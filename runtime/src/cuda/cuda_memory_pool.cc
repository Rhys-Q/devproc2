#ifdef DEVPROC2_WITH_CUDA

// MemoryPool routes through CUDADeviceAPI (DeviceAPIRegistry::Get(kDLCUDA)).
// A true block-pooling allocator is deferred to post-MVP.
// Storage and Tensor allocation already call DeviceAPI::Alloc, satisfying
// the X1 requirement that no direct cudaMalloc appears outside cuda_device_api.cc.

#endif  // DEVPROC2_WITH_CUDA
