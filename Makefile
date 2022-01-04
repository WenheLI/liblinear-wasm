CC = emcc
CXX = em++
AR = emar rcv
RANLIB = emranlib

CFLAGS = -Wall -Wconversion -O3 -fPIC
BUILD_DIR = dist/
EMCCFLAGS = -s ASSERTIONS=2 -s "EXPORT_NAME=\"Linear\"" -s MODULARIZE=1 -s DISABLE_EXCEPTION_CATCHING=0 -s NODEJS_CATCH_EXIT=0  -s WASM=1 -s ALLOW_MEMORY_GROWTH=1

BLAS_DIR = blas/
BLAS_FILES = blas/dnrm2.o blas/daxpy.o blas/ddot.o blas/dscal.o

all: wasm

newton.o: liblinear/newton.h liblinear/newton.cpp
		$(CXX) $(CFLAGS) -c liblinear/newton.cpp -o newton.o

linear.o: liblinear/linear.cpp liblinear/linear.h
		$(CXX) $(CFLAGS) -c liblinear/linear.cpp -o linear.o

blas: liblinear/blas/blas.h liblinear/blas/*.c
	mkdir $(BLAS_DIR)
	$(CC) $(CFLAGS) -c liblinear/blas/dscal.c -o $(BLAS_DIR)/dscal.o
	$(CC) $(CFLAGS) -c liblinear/blas/ddot.c -o $(BLAS_DIR)/ddot.o
	$(CC) $(CFLAGS) -c liblinear/blas/daxpy.c -o $(BLAS_DIR)/daxpy.o
	$(CC) $(CFLAGS) -c liblinear/blas/dnrm2.c -o $(BLAS_DIR)/dnrm2.o
	$(AR) $(BLAS_DIR)/blas.a $(BLAS_FILES)
	$(RANLIB) $(BLAS_DIR)/blas.a

wasm: liblinear-wasm.c linear.o newton.o liblinear/linear.h blas
		rm -rf $(BUILD_DIR); 
		mkdir -p $(BUILD_DIR);
		$(CC) $(CFLAGS) liblinear-wasm.c linear.o newton.o blas/blas.a -o $(BUILD_DIR)/liblinear.js $(EMCCFLAGS)
		cp ./liblinear.d.ts $(BUILD_DIR)/liblinear.d.ts

clean: 
	rm -rf dist/
	rm -rf ./linear.o
	rm -rf ./newton.o
	rm -rf ./blas