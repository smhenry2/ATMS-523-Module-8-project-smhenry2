# ATMS-523-Module-8-project-smhenry2
Module 8 project for ATMS523 - Logistic Regression for Identifying Developing and Non-developing TCs

This module will evaluate the performace of a logistic regression model in identifying whether a tropical disturbance will develop or not.

The dataset is based off ERA5 vorticity and moisture tracking to identify TC "seeds," both developing and non-developing. The developing seeds are matched to IBTrACS to truncate the distrbance before TC genesis.

## Workflow
1. Set up
	1. Acquire Dr. Feng's raw vorticity tracks folder "seeds_TRACK" and matched TCG tracks folder "MATCH-NH-Xiangbo". 
	2. Run Miaorui's processing code (with non-developer processing added) for just the North Atlantic to create a file of all developers and a file of all non-developers
2. Run ERA5 processing for 925-hPa vorticity, 975-hPa convergence, 700-hPa specific humidity
3. Visualize the timeseries of developers versus non-developers
	- Use 6 hourly data as a simple explanation on the difference between dynamic and thermodynamic parameters in the inner region (1.5 deg box) and outer region (5 deg box)
4. Logistic regression
	1. Use processed ERA5 variables and lat, lon data as a matrix for developers and non-developers, split each of developers and non-developers cases into 80% for training (1980-2012) and 20% for testing (2013-2020). Keep only the first three/four (two?) days of each track.
		- Consider testing different years to split, or use a percentage within each year instead of across years.
		- Consider testing different thresholds of how many days/hours from the beginning of each track to keep.
			- Currently, using 16 steps/96 hours/4 days
		- Filter out non-devs with a maximum TCWV at the vortex throughout the lifetime less than 50 mm (45 mm?)
	2. Create a logistic regression model to predict TCG or no. The question is "can a logistic regression model predict TCG from the first few points of a seed's track?"
		1. Test using 6 hourly data for three days
		2. Test using 6 hourly data for four days
		3. Test using hourly data for the best of the above
		4. Test using TCWV>=45mm for the best of the above
	3. Test a decision tree model using the best of the above
