# %%
import json
import pandas as pd

# %%
path = '/tmp/evallogs/230727231155U34TMVPH_gpt-4,gpt-4-base_ballots.dev.v0.jsonl'
with open(path) as f:
    lines = [json.loads(line) for line in f.readlines()]

# %%
def create_log(sample_id):
    samples = [line for line in lines if 'sample_id' in line and line['sample_id'] == sample_id]
    samples = sorted(samples, key=lambda x: x['event_id'])

    log = {
        'transcript': '',
        'sample_id': sample_id,    
    }

    whoisit = ['voter', 'influencer']
    for i, sample in enumerate(samples):
        if 'sampling' == sample['type']:
            log['transcript'] += f"-----{whoisit[0]}#{i//2+1}-----\n\n{sample['data']['sampled'][0]}\n\n"
            whoisit[0], whoisit[1] = whoisit[1], whoisit[0]
        elif 'metrics' == sample['type']:
            log.update(sample['data'])
    
    return log

# %%
sample_ids = set([line['sample_id'] for line in lines if 'sample_id' in line])
logs = []
for sample_id in sample_ids:
    log = create_log(sample_id)
    logs.append(log)
df = pd.DataFrame(logs)
df = df.sort_values(by=['sample_id'])
df = df.reset_index(drop=True)
# %%
write_path = path.replace('.jsonl', '.csv')
df.to_csv(write_path)
print('wrote to', write_path)

# %%
