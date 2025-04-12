import time
import random
from datetime import datetime

def simulate_row():
    product_types = ['Electronics', 'Clothing', 'Books']
    row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'product_type': random.choice(product_types),
        'price': round(random.uniform(10, 500), 2),
        'clicks': random.randint(1, 100)
    }
    # print('rows----',row)
    return row

# For testing: Print 5 simulated rows.
if __name__ == "__main__":
    for _ in range(5):
        print(simulate_row())
        time.sleep(1)
