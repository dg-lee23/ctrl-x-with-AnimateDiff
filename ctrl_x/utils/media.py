import numpy as np
import torch
import torchvision.transforms.functional as vF

# ***
import PIL, PIL.Image, PIL.ImageOps
import os, tempfile, requests
from urllib.parse import unquote, urlparse
from typing import Any, Callable, List, Optional, Tuple, Union


JPEG_QUALITY = 95


def preprocess(image, processor, **kwargs):
    if isinstance(image, PIL.Image.Image):
        pass
    elif isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = vF.to_pil_image(image)
    else:
        raise TypeError(f"Image must be of type PIL.Image, np.ndarray, or torch.Tensor, got {type(image)} instead.")
    
    image = processor.preprocess(image, **kwargs)
    return image

def preprocess_video(video, processor, **kwargs):
    """
    Preprocesses video inputs using a processor for frame-by-frame preprocessing.
    
    Args:
        video: Video input, which can be:
            - A string (file path to a video).
            - A NumPy array of shape (T, H, W, C).
            - A PyTorch tensor of shape (T, C, H, W).
            - A list of frames, where each frame is a PIL.Image, np.ndarray, or torch.Tensor.
        processor: A processor (e.g., VaeImageProcessor) with a `preprocess()` method for single frames.
        **kwargs: Additional arguments to pass to the processor's preprocess method.
    
    Returns:
        A PyTorch tensor of shape (T, C, H, W), where T is the number of frames.
    """
    frames = []

    # Case 1: Video file path
    if isinstance(video, str):
        reader = imageio.get_reader(video, "ffmpeg")
        for frame in reader:
            pil_frame = PIL.Image.fromarray(frame)
            processed_frame = processor.preprocess(pil_frame, **kwargs)
            frames.append(processed_frame)

    # Case 2: NumPy array (T, H, W, C)
    elif isinstance(video, np.ndarray):
        if video.ndim != 4 or video.shape[-1] not in [1, 3, 4]:  # Check dimensions and channels
            raise ValueError(f"Invalid video array shape: {video.shape}. Expected (T, H, W, C).")
        for frame in video:
            pil_frame = PIL.Image.fromarray(frame)
            processed_frame = processor.preprocess(pil_frame, **kwargs)
            frames.append(processed_frame)

    # Case 3: PyTorch tensor (T, C, H, W)
    elif isinstance(video, torch.Tensor):
        if video.ndim != 4 or video.shape[1] not in [1, 3]:  # Check dimensions and channels
            raise ValueError(f"Invalid video tensor shape: {video.shape}. Expected (T, C, H, W).")
        for frame in video:
            pil_frame = vF.to_pil_image(frame)
            processed_frame = processor.preprocess(pil_frame, **kwargs)
            frames.append(processed_frame)

    # Case 4: List of frames
    elif isinstance(video, list):
        for frame in video:
            if isinstance(frame, PIL.Image.Image):
                pil_frame = frame
            elif isinstance(frame, np.ndarray):
                pil_frame = PIL.Image.fromarray(frame)
            elif isinstance(frame, torch.Tensor):
                pil_frame = vF.to_pil_image(frame)
            else:
                raise TypeError(f"Frame must be PIL.Image, np.ndarray, or torch.Tensor, got {type(frame)}.")
            
            processed_frame = processor.preprocess(pil_frame, **kwargs)
            frames.append(processed_frame)
    else:
        raise TypeError(f"Video must be a file path, np.ndarray, torch.Tensor, or list of frames, got {type(video)}.")

    # stack into (f, b=1, c, h, w)
    ret = torch.stack(frames)   

    # squueze to (f, c, h, w)
    return ret.squeeze(dim=1)

def is_imageio_available():
    import importlib.util
    return importlib.util.find_spec("imageio") is not None


from typing import List, Optional, Callable
import os
import tempfile
import requests
from urllib.parse import urlparse, unquote
from PIL import Image
import numpy as np

def load_video(
    video: str,
    convert_method: Optional[Callable[[List[Image.Image]], List[Image.Image]]] = None,
) -> List[Image.Image]:
    """
    Loads `video` to a list of PIL Image and preprocesses it.
    Args:
        video (`str`):
            A URL or Path to a video to convert to a list of PIL Image format.
        convert_method (Callable[[List[PIL.Image.Image]], List[PIL.Image.Image]], *optional*):
            A conversion method to apply to the video after loading it. When set to `None`, the images will be converted
            to "RGB".
    Returns:
        `List[PIL.Image.Image]`:
            The video as a list of PIL images, resized to 512x512 and capped at 16 frames.
    """
    is_url = video.startswith("http://") or video.startswith("https://")
    is_file = os.path.isfile(video)
    was_tempfile_created = False

    if not (is_url or is_file):
        raise ValueError(
            f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {video} is not a valid path."
        )

    if is_url:
        response = requests.get(video, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Failed to download video. Status code: {response.status_code}")
        parsed_url = urlparse(video)
        file_name = os.path.basename(unquote(parsed_url.path))
        suffix = os.path.splitext(file_name)[1] or ".mp4"
        video_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
        was_tempfile_created = True
        video_data = response.iter_content(chunk_size=8192)
        with open(video_path, "wb") as f:
            for chunk in video_data:
                f.write(chunk)
        video = video_path

    pil_images = []

    if video.endswith(".gif"):
        gif = Image.open(video)
        try:
            while True:
                pil_images.append(gif.copy())
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
    else:
        if is_imageio_available():
            import imageio
        else:
            raise ImportError("ImageIo is not available.")

        try:
            imageio.plugins.ffmpeg.get_exe()
        except AttributeError:
            raise AttributeError(
                "Unable to find an ffmpeg installation on your machine. Please install via `pip install imageio-ffmpeg`"
            )

        with imageio.get_reader(video) as reader:
            for frame in reader:
                pil_images.append(Image.fromarray(frame))

    if was_tempfile_created:
        os.remove(video)

    # Preprocess: Resize to 512x512 and cap to 16 frames
    # Updated resize line
    pil_images = [img.resize((512, 512), Image.Resampling.LANCZOS) for img in pil_images]


    if len(pil_images) > 16:
        indices = np.linspace(0, len(pil_images) - 1, 16, dtype=int)
        pil_images = [pil_images[i] for i in indices]

    if convert_method is not None:
        pil_images = convert_method(pil_images)

    return pil_images
