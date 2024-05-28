# nice

## Getting Started

To get started with this project, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone <repository_url>
    cd NICE
    ```

2. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Prepare the dataset:**
    Run the `prepare_data.py` script to download and preprocess the dataset.
    ```sh
    python data/prepare_data.py
    ```

4. **Train the model:**
    Use the `train.py` script to start training the model.
    ```sh
    python scripts/train.py
    ```

5. **Evaluate the model:**
    Use the `eval.py` script to evaluate the trained model.
    ```sh
    python scripts/eval.py
    ```
