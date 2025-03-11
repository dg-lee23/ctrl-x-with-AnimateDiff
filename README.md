# Ctrl-X with AnimateDiff (Text-to-Video)  

This repository is an extension of **[Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance](https://github.com/genforce/ctrl-x)**. This project adapts the original project for **text-to-video (T2V)** using **AnimateDiff v1.5.3** (https://github.com/guoyww/AnimateDiff).  

Optionally, adjust the feature_injection() method in ./ctrl_x/utils/feature.py to achieve *reframing*, which enables object shifting, camera zoom and perspective adjustment.