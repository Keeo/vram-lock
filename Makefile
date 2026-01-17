# Makefile for vram_lock.cpp (CUDA Driver API + OpenSSL)
#
# Usage:
#   make
#   make run
#   make clean
#
# Overrides:
#   make CUDA_HOME=/usr/local/cuda
#   make CXX=clang++
#   make SUPPRESS_OPENSSL_DEPRECATED=1

CXX ?= g++
CUDA_HOME ?= /opt/cuda

TARGET := vram_lock
SRC    := vram_lock.cpp

CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS := -I$(CUDA_HOME)/include
LDFLAGS  := -L$(CUDA_HOME)/lib64
LDLIBS   := -lcuda

# Optional: silence OpenSSL 3.0 MD5 deprecation warnings if you kept MD5()
ifeq ($(SUPPRESS_OPENSSL_DEPRECATED),1)
  CXXFLAGS += -Wno-deprecated-declarations
endif

.PHONY: all clean run print-vars

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

print-vars:
	@echo "CXX=$(CXX)"
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "CXXFLAGS=$(CXXFLAGS)"
	@echo "CPPFLAGS=$(CPPFLAGS)"
	@echo "LDFLAGS=$(LDFLAGS)"
	@echo "LDLIBS=$(LDLIBS)"
