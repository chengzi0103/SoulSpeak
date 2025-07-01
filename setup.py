

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt',encoding='utf-8') as requirements_file:
    all_pkgs = requirements_file.readlines()

requirements = [pkg.replace('\n', '') for pkg in all_pkgs if "#" not in pkg]
test_requirements = []

setup(
    name='soul-speak',
    author='Cheng Chen',
    author_email='chenzi00103@gmail.com,',
    description='SoulSpeak is an emotionally driven AI companion framework that brings the voice of loved ones back into your life. By combining custom speech synthesis (TTS), large language models (LLMs), and memory-based dialogue systems, SoulSpeak offers a deeply personal, soothing voice experience',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'soul=soul.cli:soul',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='soul_speak',
    packages=find_packages(where='soul_speak', include=['soul_speak', 'soul_speak.*']),
    package_dir={'': '.'},
    test_suite='tests',
    tests_require=test_requirements,
    version='0.0.1',
    zip_safe=False,
    dependency_links=[]
)







