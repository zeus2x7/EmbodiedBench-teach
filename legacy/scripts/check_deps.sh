source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

echo "Checking embench..."
conda activate embench
python -c "import cv2; print('cv2 imported')" || echo "cv2 missing in embench"
python -c "import imageio; print('imageio imported')" || echo "imageio missing in embench"

echo "Checking embench_nav..."
conda activate embench_nav
python -c "import cv2; print('cv2 imported')" || echo "cv2 missing in embench_nav"
python -c "import imageio; print('imageio imported')" || echo "imageio missing in embench_nav"

echo "Checking embench_man..."
conda activate embench_man
python -c "import cv2; print('cv2 imported')" || echo "cv2 missing in embench_man"
python -c "import imageio; print('imageio imported')" || echo "imageio missing in embench_man"
