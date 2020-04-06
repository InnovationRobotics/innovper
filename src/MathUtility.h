#pragma once

namespace IAIRoboticsAlgorithms
{
	namespace MathUtility
	{
		// Calculate the next power of 2, there is a special implementation for int version 
		template <typename T>
		inline T NearestPowerOf2(T x)
		{
			return std::pow(2, std::ceil(std::log(x) / std::log(2)));
		}

		template <>
		inline int NearestPowerOf2(int x)
		{
			--x;
			x |= x >> 1;
			x |= x >> 2;
			x |= x >> 4;
			x |= x >> 8;
			x |= x >> 16;
			return ++x;
		}

		// Calculate the inverse of 3x3 matrix 
		inline int CalcInv3x3(const double* __restrict mat, double* __restrict invMat)
		{
			//static method

			double det = mat[0] * (mat[8] * mat[4] - mat[7] * mat[5]) -
				mat[3] * (mat[8] * mat[1] - mat[7] * mat[2]) +
				mat[6] * (mat[5] * mat[1] - mat[4] * mat[2]);


			if (0.0 == det)
			{
				return -1;
			}

			invMat[0] = (mat[8] * mat[4] - mat[7] * mat[5]) / det;
			invMat[1] = (-(mat[8] * mat[1] - mat[7] * mat[2])) / det;
			invMat[2] = ((mat[5] * mat[1] - mat[4] * mat[2])) / det;
			invMat[3] = (-(mat[8] * mat[3] - mat[6] * mat[5])) / det;
			invMat[4] = (mat[8] * mat[0] - mat[6] * mat[2]) / det;
			invMat[5] = (-(mat[5] * mat[0] - mat[3] * mat[2])) / det;
			invMat[6] = (mat[7] * mat[3] - mat[6] * mat[4]) / det;
			invMat[7] = (-(mat[7] * mat[0] - mat[6] * mat[1])) / det;
			invMat[8] = (mat[4] * mat[0] - mat[3] * mat[1]) / det;

			return 0;
		}

		// Calculate the inverse of 4x4 matrix 
		inline bool CalcInv4x4(const double m[16], double invOut[16])
		{
			double inv[16], det;
			int i;

			inv[0] = m[5] * m[10] * m[15] -
				m[5] * m[11] * m[14] -
				m[9] * m[6] * m[15] +
				m[9] * m[7] * m[14] +
				m[13] * m[6] * m[11] -
				m[13] * m[7] * m[10];

			inv[4] = -m[4] * m[10] * m[15] +
				m[4] * m[11] * m[14] +
				m[8] * m[6] * m[15] -
				m[8] * m[7] * m[14] -
				m[12] * m[6] * m[11] +
				m[12] * m[7] * m[10];

			inv[8] = m[4] * m[9] * m[15] -
				m[4] * m[11] * m[13] -
				m[8] * m[5] * m[15] +
				m[8] * m[7] * m[13] +
				m[12] * m[5] * m[11] -
				m[12] * m[7] * m[9];

			inv[12] = -m[4] * m[9] * m[14] +
				m[4] * m[10] * m[13] +
				m[8] * m[5] * m[14] -
				m[8] * m[6] * m[13] -
				m[12] * m[5] * m[10] +
				m[12] * m[6] * m[9];

			inv[1] = -m[1] * m[10] * m[15] +
				m[1] * m[11] * m[14] +
				m[9] * m[2] * m[15] -
				m[9] * m[3] * m[14] -
				m[13] * m[2] * m[11] +
				m[13] * m[3] * m[10];

			inv[5] = m[0] * m[10] * m[15] -
				m[0] * m[11] * m[14] -
				m[8] * m[2] * m[15] +
				m[8] * m[3] * m[14] +
				m[12] * m[2] * m[11] -
				m[12] * m[3] * m[10];

			inv[9] = -m[0] * m[9] * m[15] +
				m[0] * m[11] * m[13] +
				m[8] * m[1] * m[15] -
				m[8] * m[3] * m[13] -
				m[12] * m[1] * m[11] +
				m[12] * m[3] * m[9];

			inv[13] = m[0] * m[9] * m[14] -
				m[0] * m[10] * m[13] -
				m[8] * m[1] * m[14] +
				m[8] * m[2] * m[13] +
				m[12] * m[1] * m[10] -
				m[12] * m[2] * m[9];

			inv[2] = m[1] * m[6] * m[15] -
				m[1] * m[7] * m[14] -
				m[5] * m[2] * m[15] +
				m[5] * m[3] * m[14] +
				m[13] * m[2] * m[7] -
				m[13] * m[3] * m[6];

			inv[6] = -m[0] * m[6] * m[15] +
				m[0] * m[7] * m[14] +
				m[4] * m[2] * m[15] -
				m[4] * m[3] * m[14] -
				m[12] * m[2] * m[7] +
				m[12] * m[3] * m[6];

			inv[10] = m[0] * m[5] * m[15] -
				m[0] * m[7] * m[13] -
				m[4] * m[1] * m[15] +
				m[4] * m[3] * m[13] +
				m[12] * m[1] * m[7] -
				m[12] * m[3] * m[5];

			inv[14] = -m[0] * m[5] * m[14] +
				m[0] * m[6] * m[13] +
				m[4] * m[1] * m[14] -
				m[4] * m[2] * m[13] -
				m[12] * m[1] * m[6] +
				m[12] * m[2] * m[5];

			inv[3] = -m[1] * m[6] * m[11] +
				m[1] * m[7] * m[10] +
				m[5] * m[2] * m[11] -
				m[5] * m[3] * m[10] -
				m[9] * m[2] * m[7] +
				m[9] * m[3] * m[6];

			inv[7] = m[0] * m[6] * m[11] -
				m[0] * m[7] * m[10] -
				m[4] * m[2] * m[11] +
				m[4] * m[3] * m[10] +
				m[8] * m[2] * m[7] -
				m[8] * m[3] * m[6];

			inv[11] = -m[0] * m[5] * m[11] +
				m[0] * m[7] * m[9] +
				m[4] * m[1] * m[11] -
				m[4] * m[3] * m[9] -
				m[8] * m[1] * m[7] +
				m[8] * m[3] * m[5];

			inv[15] = m[0] * m[5] * m[10] -
				m[0] * m[6] * m[9] -
				m[4] * m[1] * m[10] +
				m[4] * m[2] * m[9] +
				m[8] * m[1] * m[6] -
				m[8] * m[2] * m[5];

			det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

			if (det == 0)
				return false;

			det = 1.0 / det;

			for (i = 0; i < 16; i++)
				invOut[i] = inv[i] * det;

			return true;
		}

		// Test if the input data is collinear, this function mainly used with assert statement 
		template <typename T>
		inline bool IsCollinear(const T* data, int numElements)
		{
			for (auto i = 0; i < numElements - 1; i++)
			{
				if (data[i] != data[i + 1])
					return false;
			}
			return true;
		}

		// On C++ 14 use this instead 
		// constexpr double pi() { return std::atan(1) * 4; }
		template <typename T = double>
		constexpr T pi() { return 3.14159265358979323846; }
	};
};

