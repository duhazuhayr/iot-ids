import pandas as pd
import numpy as np
import os

def generate_iot_dataset(n_samples=10000, output_path='data/iot_intrusion_dataset.csv'):
    np.random.seed(42)
    
    # 1. Base Structure (Clean Normal Traffic)
    # ---------------------------------------
    data = {
        'ts': np.linspace(1610000000, 1610000000 + 86400, n_samples),
        'src_ip': np.random.choice(['192.168.1.5', '192.168.1.10'], n_samples),
        'src_port': np.random.randint(1024, 65535, n_samples),
        'dst_ip': np.random.choice(['10.0.0.1', '8.8.8.8'], n_samples),
        'dst_port': np.random.choice([80, 443, 1883], n_samples), 
        'proto': np.random.choice(['tcp', 'udp'], n_samples, p=[0.8, 0.2]),
        'service': np.random.choice(['http', 'ssl', 'mqtt'], n_samples),
        'duration': np.random.uniform(0.1, 2.0, n_samples), # Very tight range
        'src_bytes': np.random.randint(100, 500, n_samples), # Very tight range
        'dst_bytes': np.random.randint(100, 500, n_samples),
        'conn_state': np.random.choice(['SF'], n_samples),
        'missed_bytes': 0,
        'src_pkts': np.random.randint(5, 15, n_samples),
        'dst_pkts': np.random.randint(5, 15, n_samples),
        'src_ip_bytes': np.random.randint(500, 1500, n_samples),
        'dst_ip_bytes': np.random.randint(500, 1500, n_samples),
        'label': 0,
        'type': 'normal'
    }
    
    df = pd.DataFrame(data)
    
    # 2. Inject Attacks (Mathematical Extremes)
    # -----------------------------------------
    def modify_samples(mask, label, attack_type, duration_val, byte_val, port=None):
        df.loc[mask, 'label'] = label
        df.loc[mask, 'type'] = attack_type
        df.loc[mask, 'duration'] = duration_val
        df.loc[mask, 'src_bytes'] = byte_val
        if port:
            df.loc[mask, 'dst_port'] = port

    # DoS: Instant fail (duration=0, bytes=0)
    dos_mask = np.random.choice(df.index, int(n_samples * 0.1), replace=False)
    modify_samples(dos_mask, 1, 'dos', 0.001, 10, port=8888)
    
    # DDoS: Huge burst (bytes=99999)
    remaining = df.index.difference(dos_mask)
    ddos_mask = np.random.choice(remaining, int(n_samples * 0.1), replace=False)
    modify_samples(ddos_mask, 1, 'ddos', 0.001, 99999, port=9999)
    
    # Injection: Long duration payload (duration=99, bytes=50000)
    remaining = df.index.difference(np.concatenate([dos_mask, ddos_mask]))
    inj_mask = np.random.choice(remaining, int(n_samples * 0.05), replace=False)
    modify_samples(inj_mask, 1, 'injection', 99.0, 50000)

    # 3. Unseen Attacks (Extreme Anomaly for AE)
    # -----------------------------------------
    # Scanning: Random High Ports, weird pkts
    remaining = df.index.difference(np.concatenate([dos_mask, ddos_mask, inj_mask]))
    scanning_mask = np.random.choice(remaining, int(n_samples * 0.05), replace=False)
    df.loc[scanning_mask, 'label'] = 1
    df.loc[scanning_mask, 'type'] = 'scanning'
    df.loc[scanning_mask, 'dst_port'] = 6666
    df.loc[scanning_mask, 'duration'] = 0.0001
    df.loc[scanning_mask, 'src_pkts'] = 999
    
    # 4. Final Save
    # -------------
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv(output_path, index=False)
    print(f"ULTRA-CLEAN Dataset generated at {output_path}")

if __name__ == "__main__":
    generate_iot_dataset()
