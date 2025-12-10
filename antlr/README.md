# Generating C++ Files with ANTLR

This guide explains how to generate C++ files from ANTLR grammar files in the `antlr` folder.

## 1. Install the ANTLR C++ Runtime Library

### Option A: Install from Package Manager (Fedora)
Install the ANTLR C++ runtime library using the following command:
```bash
sudo dnf install antlr4-cpp-runtime-devel
```
**Note:** The package version may be older (e.g., 4.10.1) and might not match the ANTLR JAR version.

### Option B: Compile and Install from Source (Recommended)
To ensure version compatibility with the ANTLR JAR file, compile the runtime from source:

```bash
# Clone the ANTLR4 repository
cd /tmp
git clone https://github.com/antlr/antlr4.git
cd antlr4

# Checkout the matching version (e.g., 4.13.2)
git checkout 4.13.2

# Build and install the C++ runtime
cd runtime/Cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

This will install:
- Headers to `/usr/local/include/antlr4-runtime`
- Libraries to `/usr/local/lib64/libantlr4-runtime.so`

## 2. Install Java Runtime Environment
ANTLR requires Java to run. Install the Java Runtime Environment with:
```bash
sudo dnf install java-latest-openjdk
```

## 3. Download the ANTLR Tool JAR File
Download the ANTLR tool JAR file from the official website:
```bash
wget https://www.antlr.org/download/antlr-4.13.2-complete.jar -O antlr-4.13.2-complete.jar
```
Set an environment variable for the ANTLR JAR file:
```bash
export ANTLR_JAR="antlr-4.13.2-complete.jar"
```

## 4. Generate C++ Files
To generate C++ files from the `Lang.g4` grammar file, run the following command in the `antlr` folder:
```bash
java -jar $ANTLR_JAR -Dlanguage=Cpp Lang.g4 -visitor -o generated
```
This will generate the necessary C++ files in the `generated` folder.

## 5. Build the Parser Library
Use CMake to build the parser library:
```bash
mkdir build && cd build
cmake ..
make
```
This will create `liblang_parser.a` static library.

## 6. Include the Generated Files in Your Project
Make sure to include the generated files in your C++ project:
- Add the `generated` folder to your include paths
- Link the ANTLR runtime library during compilation
- Link the `liblang_parser.a` library

## Notes
- Ensure that the `antlr` folder contains the `Lang.g4` grammar file before running the generation command.
- The `-visitor` option generates visitor classes for traversing the parse tree.
- The `-o generated` option specifies the output directory for the generated files.
- **Important:** The ANTLR JAR version and C++ runtime version must match to avoid compilation errors.