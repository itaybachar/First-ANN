#include "MNISTReader.h"

void MNISTReader::loadImages(const char * filename, std::vector<std::vector<double>>& imageArray)
{
	uint32_t imageNum, rows, cols;

	_sopen_s(&fh, filename, _O_BINARY, SH_DENYWR, _O_RDONLY);
	avail = _read(fh, buf, SIZE);

	current = buf;

	//Skip the magic number (4 bytes)
	skip(4);

	//Read # of Images
	imageNum = readNextInt();

	//Read # of Rows
	rows = readNextInt();

	//Read Num of Cols
	cols = readNextInt();

	imageArray = std::vector<std::vector<double>>(imageNum, std::vector<double>(rows*cols));
	int count;

	for (int imagesRead = 0; imagesRead < imageNum; imagesRead += count) {
		count = avail / (rows * cols);
		for (int j = 0; j < count; j++) {
			for (int pixel = 0; pixel < rows*cols; pixel++) {
				imageArray[j+ imagesRead][pixel] = ((double)readNextByte());
			}
		}
		readNextChunk();
	}
	_close(fh);
}

void MNISTReader::loadLabels(const char * filename, std::vector<double>& labelArray)
{
	uint32_t labelNum;

	_sopen_s(&fh, filename, _O_BINARY, SH_DENYWR, _O_RDONLY);
	avail = _read(fh, buf, SIZE);

	current = buf;

	//Skip the magic number (4 bytes)
	skip(4);

	//Read # of Images
	labelNum = readNextInt();

	labelArray = std::vector<double>(labelNum);
	int count;

	for (int labelsRead = 0; labelsRead < labelNum; labelsRead += count) {
		count = avail;
		for (int j = 0; j < count; j++) {
			labelArray[j + labelsRead] = ((double)readNextByte());
		}
		readNextChunk();
	}
	_close(fh);
}

uint32_t MNISTReader::readNextInt()
{
	uint32_t i = *((uint32_t*)current); current += sizeof(uint32_t);
	avail -= sizeof(uint32_t);
	reverseInt(i);

	return i;
}

uint8_t MNISTReader::readNextByte()
{
	uint8_t im = *((uint8_t*)current); current += sizeof(uint8_t);
	avail -= sizeof(uint8_t);
	return im;
}

void MNISTReader::reverseInt(uint32_t & i)
{
	uint8_t byte1 = i;
	uint8_t byte2 = i>>8;
	uint8_t byte3 = i>>16;
	uint8_t byte4 = i>>24;
	i = ((uint32_t)byte1 << 24) + ((uint32_t)byte2 << 16) + ((uint32_t)byte3 << 8) + ((uint32_t)byte4);
}
