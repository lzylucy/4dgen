""" THIS IS COPY PASTAED FROM ANZU by sfeng """

from functools import partial
import inspect
import multiprocessing as mp
import os

try:
    import ray
except ModuleNotFoundError:
    ray = None


def _raise_if_not_generator(worker):
    while isinstance(worker, partial):
        worker = worker.func
    if not inspect.isgeneratorfunction(worker):
        raise RuntimeError(
            f"\n\nThe following must be a generator function using the "
            f"`yield` statement: {worker}\nFor more details, see: "
            f"https://docs.python.org/3.6/howto/functional.html#generators"
            f"\n\n"
        )


def _parallel_iwork_unordered(
    worker, values, *, process_count, pass_process_index, ctx
):
    """
    Private implementation to support `parallel_work`.
    """
    # Based on: https://docs.python.org/2/library/multiprocessing.html#examples
    # "An example showing how to use queues to feed tasks to a collection of
    # worker processes and collect the results"
    assert process_count > 0

    if ray is not None and ctx is ray:
        assert not pass_process_index
        output_iter = _parallel_iwork_unordered_ray(
            worker,
            values,
            process_count,
        )
        for output in output_iter:
            yield output
        return

    def worker_kwargs(process_index):
        if pass_process_index:
            return dict(process_index=process_index)
        else:
            return dict()

    inputs = ctx.Queue()
    for x in values:
        inputs.put(x)
    outputs = ctx.JoinableQueue()
    # Need a more creative token.
    stop = (
        (
            "__private_stop__",
            None,
        ),
    )

    def target(inputs, outputs, process_index):
        values_iter = iter(inputs.get, stop)
        kwargs = worker_kwargs(process_index)
        output_iter = worker(values_iter, **kwargs)
        for output in output_iter:
            outputs.put(output)
        # N.B. We must explicitly keep the process alive until all produced
        # items are consumed for libraries like torch (#5348).
        outputs.join()

    ps = []

    try:
        for i in range(process_count):
            p = ctx.Process(target=target, args=(inputs, outputs, i))
            ps.append(p)
            p.start()

        # Join, effectively flushing queue, possibly unordered.
        for p in ps:
            inputs.put(stop)

        # Poll processes and check who is done.
        while len(ps) > 0:
            # Flush out queue.
            while not outputs.empty():
                yield outputs.get()
                outputs.task_done()

            for p in ps:
                if p.exitcode is None:
                    pass
                elif p.exitcode == 0:
                    p.join()
                    ps.remove(p)
                else:
                    raise RuntimeError(
                        "Process died with code {}".format(p.exitcode)
                    )

        assert outputs.empty()
    finally:
        for p in ps:
            if p.is_alive():
                p.terminate()


# Optional `ray` feature.
if ray is not None:

    # Allow actors to *persist* (for slow-to-import modules).
    _ray_actors = []
    # Simple counter for when `worker` needs to be re-done.
    # N.B. I (Eric) assume that using the `is` operator may not work if
    # pickling is used to reconsistute functions across process boundaries.
    _ray_invocation = 0

    @ray.remote(num_gpus=0.01)  # Hope it round-robins...
    class _RayActor:
        def __init__(self):
            self._worker_queue = None
            self._invocation = None

        # Hoping that `worker` is quickly pickled...
        def run(self, invocation, worker, value):
            if invocation != self._invocation:
                # Reset with new work.
                self._worker_queue = _QueuedGenerator(worker)
            return self._worker_queue(value)

    def _parallel_iwork_unordered_ray(worker, values, process_count):
        global _ray_invocation, _ray_actors
        assert (
            ray.is_initialized()
        ), "Must call ray.init() explicitly before this"
        _ray_invocation += 1
        needed = process_count - len(_ray_actors)
        if needed > 0:
            _ray_actors += [_RayActor.remote() for _ in range(needed)]
        actors = _ray_actors[:process_count]
        pool = ray.util.ActorPool(actors)

        def fn(actor, value):
            return actor.run.remote(_ray_invocation, worker, value)

        return pool.map_unordered(fn, values)


class _UnitQueueIter:
    # Provides an interator which consumes and prodouces on value at a time.
    def __init__(self):
        self._next_value = None
        pass

    def put(self, value):
        assert self._next_value is None
        self._next_value = value

    def __iter__(self):
        return self

    def __next__(self):
        assert self._next_value is not None
        value = self._next_value
        self._next_value = None
        return value


# N.B. Does not work with `p.map` as the generator is not pickleable.
class _QueuedGenerator:
    # Permits converting an iterative generator (with one argument) into a
    # stateful function.
    def __init__(self, gen, **kwargs):
        self._input = _UnitQueueIter()
        self._output = gen(self._input, **kwargs)

    def __call__(self, value):
        self._input.put(value)
        return next(self._output)


def parallel_work(
    worker,
    values,
    *,
    process_count=-1,
    progress_cls=None,
    pass_process_index=False,
    async_enum=False,
    ctx=mp,
):
    """
    Processes iterable `values` given a generator which takes an iterable
    input and returns each output. Outputs will be returned in the same
    order of the respective inputs.

    While ``multiprocessing`` provides functions like ``mp.Pool.map``,
    ``.imap``, and ``.imap_unordered``, they do not offer easy mechanisms for
    *persistence* in each worker. (There is ``Pool(initializer=...)``, but
    requires some legwork to connect.)

    Examples of persistence:
     - When doing clutter gen, you want to pre-parse the (MultibodyPlant,
       SceneGraph) pairs, and then use them in the same process.
       (At present, these objects are not pickleable).

    @param worker
        Generator function, using `yield` keyword. See:
        https://docs.python.org/3.6/howto/functional.html#generators
    @param values
        Values for generator to operate on.
    @param process_count
        Number of CPUs; if -1 or None, use all available; if 0, use this
        process (useful for debugging / simplicity) do not use multiprocessing.
        Note that process_count=1 is distinct from process_count=0, in that it
        will spin up a separate process and require transfer via pickling.
    @param progress_cls
        Wraps the iterator in the form of `progress_cls(it, total=len(values)`
        (e.g. ``tqdm.tqdm`` or ``tqdm.tqdm_notebook``).
    @param pass_process_index
        Pass index of given process (0...process_count-1) as
        ``worker(..., process_index=x)``. This can be used to help delegate
        work to certain GPUs.
    @param async_enum
        If True, will pass back an iterator, which will yield (index, output)
        rather than just output.
    @param ctx
        The multiprocessing context to use. Default is mp (multiprocessing)
        itself, but it can also be:
        - mp.dummy - to use threading instead
        - mp.get_context(method) to use a different start method (e.g. "fork")
        - torch.multiprocessing to use torch's lightweight customizations
        (though for Python 3.6+, torch registers its type-specific hooks in mp
        itself).
        - ray to use ray as the execution engine, and keep *persistent* actors
        (i.e., less time re-importing libraries). See `process_util_ray_test`
        for example.

    @warning At present, since `parallel_work` uses local nested functions with
        closures (which cannot be pickled), the "spawn" method cannot yet be
        used.

    Example:

        import multiprocessing as mp
        import random
        import time

        def worker(values):
            # Count how much work each worker does.
            count = 0
            for value in values:
                time.sleep(random.uniform(0.05, 0.1))
                count += 1
                yield (mp.current_process().name, count, value)

        outputs = parallel_work(worker, range(5), process_count=3))

    Sample contents of `outputs`, with `process_count=3`:

        [('Process-1', 1, 0),
         ('Process-2', 1, 1),
         ('Process-3', 1, 2),
         ('Process-2', 2, 3),
         ('Process-1', 2, 4)]

    With `process_count=0`:

         [('MainProcess', 1, 0),
         ('MainProcess', 2, 1),
         ('MainProcess', 3, 2),
         ('MainProcess', 4, 3),
         ('MainProcess', 5, 4)]
    """

    _raise_if_not_generator(worker)
    n = len(values)
    if not hasattr(values, "__getitem__"):
        raise RuntimeError(f"`values` must be indexable: {type(values)}")
    if process_count is None or process_count == -1:
        process_count = os.cpu_count()

    def wrap(it):
        if progress_cls:
            return progress_cls(it, total=n)
        else:
            return it

    if process_count != 0:

        def worker_wrap(indices, **kwargs):
            # Preserve iteration using queued generator so that we can monitor
            # its progression using the `progress_cls` wrapper.
            worker_queued = _QueuedGenerator(worker, **kwargs)
            for i in indices:
                output = worker_queued(values[i])
                yield (i, output)

        indices = range(len(values))
        enum_outputs_iter = _parallel_iwork_unordered(
            worker_wrap,
            indices,
            process_count=process_count,
            pass_process_index=pass_process_index,
            ctx=ctx,
        )
        enum_outputs_iter = wrap(enum_outputs_iter)

        if async_enum:
            return enum_outputs_iter

        enum_outputs = list(enum_outputs_iter)
        assert len(enum_outputs) == n, len(enum_outputs)
        # Re-order.
        outputs = n * [None]
        for i, output in enum_outputs:
            outputs[i] = output
        return outputs
    else:
        kwargs = dict()
        if pass_process_index:
            kwargs = dict(process_index=0)
        outputs_iter = wrap(worker(values, **kwargs))

        if async_enum:
            return enumerate(outputs_iter)

        return list(outputs_iter)
