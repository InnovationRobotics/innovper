#pragma once

// This is used to suppress windows.h min and max function which comes into conflict with std::numeric_limits<T>::min() functions
#ifndef NOMINMAX
	#define NOMINMAX
#endif
#ifdef min
	#undef min
#endif
#ifdef max
	#undef max
#endif

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <limits>
#include <assert.h>
#include <experimental/filesystem>
#include <assert.h>
#include "FileUtility.h"
//#include "IAIRoboticsAlgorithms_export.h"


namespace IAIRoboticsAlgorithms
{
	class /*IAIROBOTICSALGORITHMS_EXPORTS_API*/ BinaryIO
	{
	public:
	
		static bool WriteBinary(const void* src, int numBands, int numBytePerBand, int srcWidth,
		 	int srcHeight, const std::string& saveFolder, const std::string& fileName )
		{
			// const char* savePath, std::string& stringSavePath
			std::vector<unsigned char> data;
			auto sizeByte = numBands * numBytePerBand * srcWidth * srcHeight;
			data.resize(sizeByte);

			memcpy(&data[0], src, sizeByte);

			std::string stringSavePath = saveFolder + fileName + 
				BinaryIO::GetBinaryToken(numBands, numBytePerBand, srcWidth, srcHeight) + ".bin";

			// Save the raw data 
			std::ofstream file;
			file.open(stringSavePath, std::ios::out | std::ios::binary);

			if (!file.is_open()) return false;

			file.write((char*)&data[0], sizeByte);
			file.close();

			return true;
		}

		static bool ReadBinary(const char* binaryImgPath, int& numBands, int& numBytePerBand, int& width, int& height, std::vector<unsigned char>& data)
		{
			//assert(std::experimental::filesystem::exists(binaryImgPath));

			std::string pathstr, name, ext, token = "$$";
			IAIRoboticsAlgorithms::FileUtility::GetFileParts(std::string(binaryImgPath), pathstr, name, ext);

			// Substring the token from file name 
			auto found = name.find(token);
			if (found == std::string::npos) return -1;

			auto relaventName = name.substr(found + token.length(), name.length() - found);

			auto splitedData = IAIRoboticsAlgorithms::StringUtility::SplitString(relaventName, "_");

			// An example of file name levena_2_w5120_h5120.bin
			numBands = std::numeric_limits<int>::min();
			numBytePerBand = std::numeric_limits<int>::min();
			width = std::numeric_limits<int>::min();
			height = std::numeric_limits<int>::min();

			for (auto& st : splitedData)
			{
				if (st.size())
				{
					if (width == std::numeric_limits<int>::min() && st.at(0) == 'w')
					{
						width = GetNumberFromString(st);
					}
					else if (height == std::numeric_limits<int>::min() && st.at(0) == 'h')
					{
						height = GetNumberFromString(st);
					}
					else if (numBands == std::numeric_limits<int>::min() && st.at(0) == 'c')
					{
						numBands = GetNumberFromString(st);
					}
					else if (numBytePerBand == std::numeric_limits<int>::min())
					{
						if (st.at(0) == 'f')
						{
							numBytePerBand = sizeof(float);
						}
						else if (st.at(0) == 'd')
						{
							numBytePerBand = sizeof(double);
						}
						else if (st.at(0) == 'u')
						{
							numBytePerBand = sizeof(unsigned char);
						}
						else if (st.at(0) == 's')
						{
							numBytePerBand = sizeof(short);
						}
					}		
					else
					{
						assert(false);
						return false;
					}
				}
			}

			if ((numBands == std::numeric_limits<int>::min()) || (numBytePerBand == std::numeric_limits<int>::min()) || (width == std::numeric_limits<int>::min()) || (height == std::numeric_limits<int>::min()))
				return false; 

			auto dataLenghtByte = numBands * numBytePerBand * width * height;
			assert(dataLenghtByte >= 0);
			data.resize(dataLenghtByte);

			std::ifstream file;
			file.open(binaryImgPath, std::ios::in | std::ios::binary);

			if (!file.is_open()) return false;

			file.seekg(0, std::ios::beg);
			std::streampos size = file.tellg();
			file.read((char*)&data[0], dataLenghtByte);
			file.close();

			return true;
		}

	private:
		// Get the representing token for read write binary image
		static std::string GetBinaryToken(int numBands, int bytePerPixel, int w, int h)
		{
			std::string token = "$$c" + std::to_string(numBands) + "_";
			if (bytePerPixel == sizeof(unsigned char))
			{
				token += "u";
			}
			else if (bytePerPixel == sizeof(float))
			{
				token += "f";
			}
			else if (bytePerPixel == sizeof(double))
			{
				token += "d";
			}
			else if (bytePerPixel == sizeof(unsigned short))
			{
				token += "s";
			}
			else assert(false);

			token += "_w" + std::to_string(w) + "_h" + std::to_string(h);
			return token;
		}

		static int GetNumberFromString(const std::string& st)
		{
			std::string subString = st.substr(1, st.length() - 1);
			auto num = std::stoi(subString);
			return num;
		}
	};
};