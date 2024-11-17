rm -rf apex
pip uninstall apex
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
cd ..
