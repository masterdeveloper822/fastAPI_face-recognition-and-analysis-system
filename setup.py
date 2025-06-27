from setuptools import setup, find_packages

setup(
    name="Face_Recognition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.17.0",
        "opencv-python==4.10.0.84",
        "matplotlib==3.9.2",
        "uvicorn==0.30.6",
        "python-multipart==0.0.9",
        "fastapi==0.115.0",
        "face-recognition==1.3.0",
        "fer==22.5.1",
        "dlib==19.24.6"
        
    ],

)