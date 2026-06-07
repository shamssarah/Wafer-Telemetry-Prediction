
## C++ Stream Emitter

A lightweight C++ process that reads pre-generated telemetry and emits it row by row to the FastAPI backend, simulating a real sensor on the factory floor.

### Prerequisites
- MSYS2 with UCRT64 toolchain
- vcpkg with curl installed
- CMake

### Build
```bash
cd emitter
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
```

### Run
```bash
./emitter.exe <path_to_csv> <rate_ms> <chamber_id>

# example — emit one reading every 100ms from chamber C1
./emitter.exe ../data/synthetic/telemetry_stream.csv 100 C1
```

The emitter streams single readings to `/ingest/stream`. The backend accumulates 200 readings per chamber, triggers inference, then resets. Processed sequences are archived to `data/streams/archive/`.