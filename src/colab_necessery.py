# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Copy the .rar archive from Drive to local Colab disk
!cp "/content/drive/MyDrive/data_drone/drone_audio_detector.rar" /content/

# 3. Install unrar (only takes a few seconds)
!apt-get install -y unrar

# 4. Extract the .rar archive into /content/
!unrar x /content/drone_audio_detector.rar /content/

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Copy the .rar archive from Drive to local Colab disk
!cp "/content/drive/MyDrive/data_drone/membo_phase.rar" /content/

# 3. Install unrar (only takes a few seconds)
!apt-get install -y unrar

# 4. Extract the .rar archive into /content/
!unrar x /content/membo_phase.rar /content/

!pip install librosa torch