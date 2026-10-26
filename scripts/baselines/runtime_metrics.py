from datetime import datetime
import json
from pathlib import Path

def new_usage():
    return dict(agent_input_tokens=0,agent_output_tokens=0,summary_input_tokens=0,summary_output_tokens=0)

def add_agent_usage(usage,output):
    usage['agent_input_tokens'] += len(output.prompt_token_ids)
    usage['agent_output_tokens'] += len(output.outputs[0].token_ids)

def add_summary_usage(usage,metadata):
    usage['summary_input_tokens'] += int(metadata['prompt_tokens'])
    usage['summary_output_tokens'] += int(metadata['completion_tokens'])

def finish_usage(usage):
    return dict(usage,total_input_tokens=usage['agent_input_tokens']+usage['summary_input_tokens'],total_output_tokens=usage['agent_output_tokens']+usage['summary_output_tokens'])

def assistant_output(messages):
    return '\n'.join(m['content'] for m in messages if m['role']=='assistant')

def progress(phase,**fields):
    p=Path('progress.json');q=p.with_suffix('.tmp')
    q.write_text(json.dumps(dict(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**fields),indent=2)+'\n');q.replace(p)
