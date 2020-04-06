#pragma once

#include <mutex>
#include <vector>
#include <list>
#include <thread>
#include <string>
#include "BinaryIO.h"


namespace IAIRoboticsAlgorithms
{
	class /*IAIROBOTICSALGORITHMS_EXPORTS_API*/ AsynchronousDumper
	{
	public: 
		// The class is asynchronous dump into file system a group of memory
		AsynchronousDumper() : m_isFinishThreadExecution(false)
		{
			// Note: First construct this object, than - create the thread and pass this to the working thread. Initializing the m_workerThread on the construction list causes an exception because the thread is created before the class instance finishes. 
			m_workerThread = std::thread(&AsynchronousDumper::WorkingThreadWritesDataToFileSystem, this);
		};

		virtual ~AsynchronousDumper() 
		{
			// Finalize the thread execution 
			m_isFinishThreadExecution = true; 

			std::unique_lock<std::mutex> lk(m_threadMutex);
			m_threadSyncConditionVar.notify_one();
			lk.unlock();

			m_workerThread.join();
		};

		// Specify the format of the data to write to the file system
		class DataForWrite
		{
		public:
			const void* data; 
			int numBands, numBytePerBand, srcWidth, srcHeight;
			std::string savePath;
			std::string& stringSavePath = std::string();
		};

	private:
		// Hold the data for write
		class DataForWriteInternal
		{
		public:
			DataForWriteInternal(const DataForWrite& dataForWrite)
			{
				auto dataLenByte = dataForWrite.numBands * dataForWrite.numBytePerBand * dataForWrite.srcWidth * dataForWrite.srcHeight;
				data.resize(dataLenByte);
				std::memcpy(&data[0], dataForWrite.data, dataLenByte);

				numBands = dataForWrite.numBands;
				numBytePerBand = dataForWrite.numBytePerBand;
				srcWidth = dataForWrite.srcWidth;
				srcHeight = dataForWrite.srcHeight;
				savePath = dataForWrite.savePath;
				stringSavePath = dataForWrite.stringSavePath;
			}

			std::vector<char> data;
			int numBands, numBytePerBand, srcWidth, srcHeight;
			std::string savePath;
			std::string stringSavePath;
		};
	public: 

		// The function deep copy the input data, Then the function writes the data to the file system on a different thread 
		bool WriteToFileSystem(const DataForWrite& dataForWrite)
		{
			// Deep copy the dataset 
			auto dataForListDeepCopy = std::make_unique<DataForWriteInternal>(DataForWriteInternal(dataForWrite));

			// Insert to the list 
			std::unique_lock<std::mutex> lock(m_listMutex);
			m_listOfDataToWrite.push_back(std::move(dataForListDeepCopy));
			lock.unlock();

			// Call the working thread 

			std::unique_lock<std::mutex> ul(m_threadMutex, std::try_to_lock);
			if (ul.owns_lock())
				m_threadSyncConditionVar.notify_one();

			return true;
		}

		// The function return the number of files remaining to save to the file system 
		int GetNumFilesToWrite()
		{
			return m_listOfDataToWrite.size();
			//std::string st = "[PcClassification][Log remaining files to write to file system " + std::to_string(m_asynchronousDumper->GetNumFilesToWrite()) +
			//	"][Total files " + std::to_string(m_debugCounter * m_debugNumFilesForEachRun) + "]";
			//LOGDEBUG(st.c_str());
		}

	private: 
		void WorkingThreadWritesDataToFileSystem()
		{
			//std::this_thread::sleep_for(std::chrono::milliseconds(1000));

			while (!m_isFinishThreadExecution)
			{
				// wait until the main class sends the data 
				std::unique_lock<std::mutex> lk(m_threadMutex);
				m_threadSyncConditionVar.wait(lk);

				if (!m_isFinishThreadExecution)
				{
					while (m_listOfDataToWrite.size())
					{
						// Get data from the list and write it
						std::unique_lock<std::mutex> lock(m_listMutex);
						auto dataForWrite = std::move(m_listOfDataToWrite.front());
						// Remove this note from the list 
						m_listOfDataToWrite.pop_front();
						lock.unlock();

						IAIRoboticsAlgorithms::BinaryIO::WriteBinary((void*)&dataForWrite->data[0], dataForWrite->numBands, dataForWrite->numBytePerBand,
							dataForWrite->srcWidth, dataForWrite->srcHeight, dataForWrite->savePath.c_str(), dataForWrite->stringSavePath);
					}
				}
				// Manual unlocking is done before notifying, to avoid waking up
				// The waiting thread only to block again
				lk.unlock();
			}
		}

	private: 

		// Hold the data to write by the internal thread 
		std::list<std::unique_ptr<DataForWriteInternal>> m_listOfDataToWrite;
		// Lock for operations over the list 
		std::mutex m_listMutex; 

		std::thread m_workerThread;
		std::mutex m_threadMutex;
		std::condition_variable m_threadSyncConditionVar;
		bool m_isFinishThreadExecution; 
	};
}