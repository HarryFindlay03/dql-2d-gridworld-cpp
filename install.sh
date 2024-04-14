mkdir -p "bin"
mkdir -p "build"

if [ ! -d "include/cpp-nn/Eigen" ]; then
    echo "Eigen library not installed, program will not work!"
    echo "Please copy the Eigen header library into include/cpp-nn/"
fi

