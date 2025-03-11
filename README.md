# Ctrl-X with Text-to-Video  

This repository is an extension of **[Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance](https://github.com/genforce/ctrl-x)**. This project adapts the original one for **text-to-video (T2V)** using **AnimateDiff v1.5.3** (https://github.com/guoyww/AnimateDiff).  

Use run.py or example.sh to run the code. You may optionally perform **reframing** (which achieves shifting object, adjusting camera zoom and perspective) by choosing which feature_injection() method (see ./utils/feature.py) to use.