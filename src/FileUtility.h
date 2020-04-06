#pragma once

#include <string>
#include <vector>


namespace IAIRoboticsAlgorithms
{
	class FileUtility
	{
	public:
		// Parts of file name and path. synonymous to matlab function [pathstr,name,ext] = fileparts(filename) 
		// returns the path name, file name, and extension for the specified file. The file does not have to exist. filename is a string enclosed in single quotes. The returned ext field contains a dot (.) before the file extension
		// For example for input: "C:\sdf\sdf\sdfsdf.bmp" the output would be: "C:\sdf\sdf", "sdfsdf", ".bmp"
		//	static void GetFileParts(const std::string& src, std::string& pathstr = std::string(), std::string& name = std::string(), std::string& ext = std::string());
		static void GetFileParts(const std::string& src, std::string& pathstr, std::string& name, std::string& ext)
		{
			// Initialize
			pathstr = name = ext = "";

			// Calculate path string 
			auto pathLoc = src.find_last_of("/");
			if (pathLoc != std::string::npos)
				pathstr = src.substr(0, pathLoc);

			// Calculate name and file extension
			auto extensionLoc = src.find_last_of(".");
			if (extensionLoc != std::string::npos)
			{
				auto pathLocStartPos = pathLoc + 1;
				if (pathLoc != std::string::npos &&  pathLocStartPos < src.length() && extensionLoc < src.length())
				{
					name = src.substr(pathLocStartPos, extensionLoc - pathLocStartPos);
				}
				ext = src.substr(extensionLoc, src.length());
			}
		}
	};

	class StringUtility
	{
	public:
		static std::vector<std::string> SplitString(const std::string& srcString, const std::string& delimiter)
		{
			std::vector<std::string> res;

			auto str = srcString;
			size_t pos = 0;
			while ((pos = str.find(delimiter)) != std::string::npos) {
				auto token = str.substr(0, pos);
				if (token != "")
					res.push_back(token);
				str.erase(0, pos + delimiter.length());
			}

			// push the last part of the string 
			if (res.size() && str != "")
			{
				res.push_back(str);
			}

			return res;
		}

		// Trim start and end spaces from an input string. for example for the input "   ABC   " the output would be "ABC"
		static std::string TrimStartAndEndSpaces(const std::string& str)
		{
			size_t first = str.find_first_not_of(' ');
			if (std::string::npos == first)
			{
				return str;
			}
			size_t last = str.find_last_not_of(' ');
			return str.substr(first, (last - first + 1));
		}
	};
};