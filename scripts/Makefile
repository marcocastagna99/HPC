CC = icx
CFLAGS = -g -O3 -xHost -fiopenmp
DEBUG_FLAGS = -DDEBUG

TARGET = mat_mul_p
DEBUG_TARGET = mat_mul_debug

all: $(TARGET)

$(TARGET): ../mat_mul_parallel.c
	$(CC) $(CFLAGS) -o $(TARGET) ../mat_mul_parallel.c
	@echo "Performance build complete: ./$(TARGET)"

debug: ../mat_mul_parallel.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) -o $(DEBUG_TARGET) ../mat_mul_parallel.c
	@echo "Debug build complete: ./$(DEBUG_TARGET)"

clean:
	rm -f $(TARGET) $(DEBUG_TARGET) mat-res.txt
	@echo "Cleaned up executable files."