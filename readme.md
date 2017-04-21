This is a simple even odd transposition sort using the Cuda Architecture

Right Now the implementation without shared memory seems to perform faster
than the one with shared memory.

It seems that there is no easy way to make it work with shared memory efficiently
because there will always be a problem with racing conditions between threads and
block synchronization.
