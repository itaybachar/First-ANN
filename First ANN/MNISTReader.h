#pragma once
#include <io.h>
#include <iostream>
#include <sys/stat.h>
#include <fcntl.h>
#include <vector>

const int SIZE = 32768; //Chunk read size
const int preBufSize = 256;

class MNISTReader {
private:
	char * preBuf = new char[preBufSize + SIZE];
	char * buf = preBuf+preBufSize;
	char * current;
	size_t avail;
	int fh;

	void readNextChunk() {
		//Place remaining data into prebuf
		for (uint32_t i = 0; i < avail; i++) {
			preBuf[preBufSize - avail] = current[i];
		}
		//Setting the current pos into preBuf
		current = buf - avail;
		//Reading a new chunk into the buffer
		avail += _read(fh, buf, SIZE);
	}

	void reverseInt(uint32_t& i);

	uint32_t readNextInt();

	uint8_t readNextByte();

	void skip(int amnt) {
		current += amnt;
		avail -= amnt;
	}

public:
	MNISTReader() {  };

	void loadImages(const char* filename, std::vector<std::vector<double>>& imageArray);

	void loadLabels(const char* filename, std::vector<double>& labelArray);
};