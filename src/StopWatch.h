#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>


namespace IAIRoboticsAlgorithms
{
	template < typename TimeUnit = std::chrono::milliseconds, typename Clock = std::chrono::high_resolution_clock>
	class StopWatch
	{
	public: 
		explicit StopWatch() noexcept
		{
			m_elapseTimeHistory.reserve(128);
		}

		~StopWatch() noexcept
		{

		}

		// For singleton usage 
		static StopWatch& GetInstance()
		{
			static StopWatch instance;
			return instance;
		}

		void Start() noexcept
		{
			m_start = Clock::now();

		}
		void Stop() noexcept
		{
			m_stop = Clock::now();

			auto elapseTime = std::chrono::duration_cast<TimeUnit>(m_stop - m_start);
			m_elapseTimeHistory.push_back(elapseTime.count());
		}

		void Reset() noexcept
		{
			m_elapseTimeHistory.resize(0);
		}

		void PrintRunningTimeSummery() 
		{
			if (m_elapseTimeHistory.size())
			{
				int counter = 0;
				for (auto i : m_elapseTimeHistory)
				{
					std::cout << "[Iteration " << counter++ << "][Running Time " << i << "]\n";
				}

				auto maxElement = *std::max_element(m_elapseTimeHistory.begin(), m_elapseTimeHistory.end());
				auto minElement = *std::min_element(m_elapseTimeHistory.begin(), m_elapseTimeHistory.end());

				auto sum = std::accumulate(m_elapseTimeHistory.begin(), m_elapseTimeHistory.end(), 0.);
				auto average = sum / double(m_elapseTimeHistory.size());
			
				// calculate std dev
				std::vector<double> diff(m_elapseTimeHistory.size());
				std::transform(m_elapseTimeHistory.begin(), m_elapseTimeHistory.end(), diff.begin(), [average](double x) {return x - average; });
				double sqSum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.);
				double stdDev = std::sqrt(sqSum / m_elapseTimeHistory.size());
	
				// This will modify the array of values 
				std::nth_element(m_elapseTimeHistory.begin(), m_elapseTimeHistory.begin() + m_elapseTimeHistory.size() / 2, m_elapseTimeHistory.end());
				auto median = *std::next(m_elapseTimeHistory.begin(), m_elapseTimeHistory.size() / 2);

				std::cout << "[Summery][Average " << average << "][StdDev " << stdDev << "][Median "  << median << "][Min " << minElement << "][Max " << maxElement << "][Sum " << sum << "]\n";
			}
		}

	private:
		std::vector<double> m_elapseTimeHistory;
		std::chrono::time_point<Clock> m_stop;
		std::chrono::time_point<Clock> m_start;
	};
}

