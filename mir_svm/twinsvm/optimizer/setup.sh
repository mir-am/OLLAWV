# Use one of the following arguments to generate C++ extension module
# -arma : Armadillo library is installed on the system.
# -no_arma : Armaidllo lib is NOT installed. Therefore, arma repo. will be cloned.

if [ $1 == "-arma" ]
then

c++ -O3 -Wall -shared -std=c++11 -fPIC -o ../clippdcd`python3-config --extension-suffix` ./pybind_clippdcd.cpp `python3 -m pybind11 --includes` -larmadillo -lblas -llapack

elif [ $1 == "-no_arma" ] 
then

	if [ -d "armadillo-code" ]
	then
		echo "Found Armadillo repository. No need to clone again."		
	else
		# clones Armadillo which is a C++ Linear Algebra library
		# Armadillo is licensed under the Apache License, Version 2.0
		git clone https://github.com/mir-am/armadillo-code.git
	fi

c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` ./pybind_clippdcd.cpp -o ./extensions/clippdcd`python3-config --extension-suffix` -I ./armadillo-code/include -DARMA_DONT_USE_WRAPPER -lblas -llapack 

fi
