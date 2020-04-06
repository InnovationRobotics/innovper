#pragma once

#include <matrix.h>
#include <mclcppclass.h>


template <int VoxelDim = 30, int NumClasses = 5>
class VoxelUtility
{
public:
	template <typename T = float>
	static void BuildVoxelFromXYZ(const int* x, const int* y, const int* z, const T* intensity, int numElements, T voxelMat[VoxelDim][VoxelDim][VoxelDim])
	{
		auto voxelSizeByte = VoxelDim * VoxelDim * VoxelDim * sizeof(T);
		memset(voxelMat, 0x00, sizeof(voxelMat));

		for (auto i = 0; i < numElements; i++)
		{
			voxelMat[*(z + i) - 1][*(y + i) - 1][*(x + i) - 1] = *(intensity + i) + 1;
		}
	}


	template <typename T = float, mxClassID classID = mxClassID::mxSINGLE_CLASS>
	static void BuildVoxelFromXYZ(const int* x, const int* y, const int* z, const T* intensity, int numElements, mwArray& voxelDataMwArray)
	{
		T voxelMat[VoxelDim][VoxelDim][VoxelDim];
		BuildVoxelFromXYZ<T>(x, y, z, intensity, numElements, voxelMat);

		mwSize num_dims = 3;
		const mwSize dims[3] = { VoxelDim , VoxelDim , VoxelDim };
		voxelDataMwArray = mwArray(num_dims, dims, classID);
		voxelDataMwArray.SetData(&voxelMat[0][0][0], VoxelDim * VoxelDim * VoxelDim);
	}


	template <typename T = float>
	static bool ReadVoxelFromFile(const std::string& voxelPath, std::vector<int>& x, std::vector<int>& y, std::vector<int>& z, std::vector<T>& intensity)
	{
		// Read xyz from file
		ifstream file(voxelPath.c_str());
		if (!file.is_open()) return false;

		int numLines = 2048;
		x.reserve(numLines); y.reserve(numLines); z.reserve(numLines); intensity.reserve(numLines);

		string str;
		while (std::getline(file, str))
		{
			auto lineContent = split(str, ' ');
			assert(lineContent.size() == 4);

			auto xval = stoi(lineContent[0]), yval = stoi(lineContent[1]), zval = stoi(lineContent[2]);
			auto intensityVal = stof(lineContent[3]);

			x.push_back(xval);
			y.push_back(yval);
			z.push_back(zval);
			intensity.push_back(intensityVal);
		}

		return true;
	}


	template <typename T = float>
	static bool ReadVoxelFromFile(const std::string& voxelPath, mwArray& voxelDataMwArray)
	{
		std::vector<int> x, y, z; std::vector<T> intensity;
		auto retVal = ReadVoxelFromFile(voxelPath, x, y, z, intensity);
		BuildVoxelFromXYZ(&x[0], &y[0], &z[0], &intensity[0], x.size(), voxelDataMwArray);
		return retVal;
	}

	template <typename T = float>
	static bool ReadVoxelFromFile(const std::string& voxelPath, T voxelMat[VoxelDim][VoxelDim][VoxelDim])
	{
		std::vector<int> x, y, z; std::vector<T> intensity;
		auto retVal = ReadVoxelFromFile(voxelPath, x, y, z, intensity);
		BuildVoxelFromXYZ(&x[0], &y[0], &z[0], &intensity[0], x.size(), voxelMat);
		return retVal;
	}
};
