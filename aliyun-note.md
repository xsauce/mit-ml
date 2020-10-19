ssh -i ~/ssh/self.pub root@106.14.76.68
apt-get update && apt-get install -y git
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda install numpy matplotlib scikit-learn
git clone https://github.com/xsauce/mit-ml.git
cd mit-ml