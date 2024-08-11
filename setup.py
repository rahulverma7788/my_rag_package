from setuptools import setup, find_packages

setup(
    name='my_rag_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'flask',
        # Add other dependencies here
    ],
    extras_require={
        'dev': ['pytest', 'sphinx'],
    },
    entry_points={
        'console_scripts': [
            'my-rag-command=my_rag_package.cli:main',
        ],
    },
    author='Rahul Verma',
    author_email='rahulverma207788@gmail.com.com',
    description='A package for RAG with LangChain and hybrid search',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_rag_package',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
