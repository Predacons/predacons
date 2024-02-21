# Build Guide

To get started with compiling Predacons locally, follow these steps:

1. Clone the Predacons repository:
    ```bash
    git clone https://github.com/Predacons/Predacons.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install wheel:
    ```bash
    pip install wheel
    ```

4. Build the project:
    ```bash
    python setup.py bdist_wheel sdist
    ```

5. Install Predacons:
    ```bash
    pip install .
    ```

6. Import Predacons and use it in your code:
    ```python
    from predacons import predacons
    predacons.rollout()
    ```