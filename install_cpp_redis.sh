#!/bin/bash
# Script to install cpp_redis library

set -e

echo "=========================================="
echo "Installing cpp_redis library"
echo "=========================================="

# Check if already installed
if [ -d "/usr/local/include/cpp_redis" ] || [ -d "/usr/include/cpp_redis" ]; then
    echo "✓ cpp_redis appears to be already installed"
    echo "  Checking include directory..."
    if [ -f "/usr/local/include/cpp_redis/cpp_redis.h" ] || [ -f "/usr/include/cpp_redis/cpp_redis.h" ]; then
        echo "✓ cpp_redis headers found"
        exit 0
    fi
fi

# Clone repository
if [ ! -d "cpp_redis" ]; then
    echo "Cloning cpp_redis repository..."
    git clone https://github.com/Cylix/cpp_redis.git
fi

cd cpp_redis

# Initialize and update submodules (tacopie dependency)
echo "Initializing submodules..."
git submodule init
git submodule update

# Build cpp_redis
echo "Building cpp_redis..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc 2>/dev/null || echo 4)

# Install
echo "Installing cpp_redis..."
sudo make install

# Update library cache (Linux only)
if [ "$(uname)" != "Darwin" ]; then
    echo "Updating library cache..."
    sudo ldconfig
fi

echo ""
echo "=========================================="
echo "✓ cpp_redis installed successfully!"
echo "=========================================="
echo "Include directory: /usr/local/include"
echo "Library directory: /usr/local/lib"
echo ""
echo "You can now run: cmake .. && make"
