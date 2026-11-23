import ast
import asyncio
import os
import textwrap
import threading
import time

# This server is PyTorch-only. Prevent Transformers from importing TensorFlow
# pipeline modules, which can fail when TensorFlow is installed but incomplete.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import re

import torch
import transformers
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.cfg.constants import *
from src.vlm_utils import CIRCUIT_VIEW, VLM_PROMPT, VLM_MAX_NEW_TOKENS, GemmaCritic

app = FastAPI(title="LLM API", version="1.0")
VLM = None

BATCH_SIZE = 16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 100 * 2**30 else 8  # num of LLM requests to process at once
BATCH_WAIT_TIME = 2  # max wait time for batch to fill in s

STOP_AT_FIRST_CODE_BLOCK = os.getenv("LLMGE_STOP_AT_CODE_BLOCK", "1") == "1"
CODE_BEFORE_END = os.getenv("LLMGE_CODE_BEFORE_END", "1") == "1"
SPLIT_LONG_BATCHES = os.getenv("LLMGE_SPLIT_LONG_BATCHES", "1") == "1"
LONG_PROMPT_MARKER = "This block is long."
LONG_MAX_WAIT = float(os.getenv("LLMGE_LONG_MAX_WAIT", "600"))


def pick_batch(waiting, now, batch_size, split=SPLIT_LONG_BATCHES, max_wait=LONG_MAX_WAIT):
    if not split:
        return list(range(min(batch_size, len(waiting))))
    longs = [i for i, (req, _, _) in enumerate(waiting) if LONG_PROMPT_MARKER in req["prompt"]]
    shorts = [i for i in range(len(waiting)) if i not in set(longs)]
    if longs and (len(longs) >= batch_size or not shorts or now - waiting[longs[0]][2] >= max_wait):
        return longs[:batch_size]
    return shorts[:batch_size]
_LANG_TAG = re.compile(r"\s*[A-Za-z0-9_+-]*\s*")


def parses_as_code(body):
    try:
        tree = ast.parse(textwrap.dedent(body))
    except (SyntaxError, ValueError):
        return False
    return any(not (isinstance(node, ast.Expr) and isinstance(node.value, (ast.Name, ast.Constant)))
               for node in tree.body)


def closed_code_block(text):
    parts = text.split("```")
    for seg in parts[1:-1]:
        lines = seg.split("\n")
        if lines and _LANG_TAG.fullmatch(lines[0]):
            lines = lines[1:]
        body = "\n".join(lines).strip()
        if body.count("\n") >= 1 and parses_as_code(body):
            return True
    return False


def has_code(text):
    if closed_code_block(text):
        return True
    body = "\n".join(line for line in text.split("\n") if not line.strip().startswith("```")).strip()
    return body.count("\n") >= 1 and parses_as_code(body)


class CodeBeforeEnd(LogitsProcessor):
    def __init__(self, tokenizer, end_ids, every=8):
        self.tokenizer, self.every = tokenizer, every
        self.end_ids = sorted({int(i) for i in end_ids if i is not None})
        self.start, self.calls, self.ok = None, 0, None

    def __call__(self, input_ids, scores):
        rows = input_ids.shape[0]
        if self.start is None or self.ok is None or len(self.ok) != rows or input_ids.shape[1] < self.start:
            self.start, self.calls = input_ids.shape[1], 0
            self.ok = torch.zeros(rows, dtype=torch.bool, device=scores.device)
        self.calls += 1
        if self.calls % self.every == 0:
            texts = self.tokenizer.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
            for i, text in enumerate(texts):
                if not self.ok[i] and has_code(text):
                    self.ok[i] = True
        blocked = (~self.ok).nonzero(as_tuple=True)[0]
        if len(blocked) and self.end_ids:
            ends = torch.tensor(self.end_ids, device=scores.device)
            scores[blocked.unsqueeze(1), ends.unsqueeze(0)] = -float("inf")
        return scores


def runaway(text, max_imports=30, max_repeats=12):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if sum(line.startswith(("import ", "from ")) for line in lines) >= max_imports:
        return True
    counts = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    return len(lines) > 40 and max(counts.values()) >= max_repeats


class FirstCodeBlock(StoppingCriteria):
    def __init__(self, tokenizer, every=8):
        self.tokenizer, self.every, self.start, self.calls = tokenizer, every, None, 0

    def __call__(self, input_ids, scores, **kwargs):
        if self.start is None:
            self.start = input_ids.shape[1] - 1
        self.calls += 1
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if self.calls % self.every:
            return done
        texts = self.tokenizer.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
        for i, text in enumerate(texts):
            done[i] = closed_code_block(text) or runaway(text)
        return done


class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = LLM_MAX_NEW_TOKENS
    top_p: float = 0.8
    temperature: float = 0.7

class CritiqueRequest(BaseModel):
    image_path: str
    prompt: str = VLM_PROMPT
    max_new_tokens: int = VLM_MAX_NEW_TOKENS

class LLMModel:
    _instance = None
    _lock = threading.Lock()

    @staticmethod
    def _log_cuda_state():
        print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
        print(f"torch version={torch.__version__}", flush=True)
        try:
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count()
        except Exception as err:
            print(f"torch cuda introspection failed={err}", flush=True)
            return

        print(f"torch cuda available={cuda_available}", flush=True)
        print(f"torch cuda device count={device_count}", flush=True)

    @staticmethod
    def _validate_cuda_available():
        LLMModel._log_cuda_state()
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(
                "CUDA is not available to PyTorch, so the LLM server would run on CPU. "
                "Check the Slurm GPU request, CUDA module, and uv/torch environment."
            )
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print(f"Loading model at {MODEL_PATH} for the first time", flush=True)
                    cls._instance = super(LLMModel, cls).__new__(cls)
                    cls._instance._initialize()
                    print('I created my instance', flush=True)
                    print(dir(cls._instance), flush=True)            
        return cls._instance
    
    def _initialize(self):
        print("initializing", flush=True)
        self._validate_cuda_available()
        # TODO figure out how to better handle the initialization (i.e. mixtral dies because it doesn't have attention)
        # TODO find out why when this dies the code around it continues i.e. a model is returned to generate_text, but I never see the print out of "I created my instance"
        # Resolve HF cache directories: if MODEL_PATH is a HF cache dir
        # (contains 'snapshots/'), find the actual model snapshot
        model_path = MODEL_PATH
        import os
        snapshots_dir = os.path.join(model_path, 'snapshots')
        if os.path.isdir(snapshots_dir):
            # Use the latest snapshot
            snapshot_dirs = sorted(os.listdir(snapshots_dir))
            if snapshot_dirs:
                model_path = os.path.join(snapshots_dir, snapshot_dirs[-1])
                print(f"Resolved HF cache to snapshot: {model_path}")

        device_count = torch.cuda.device_count()
        max_memory = {idx: "120GiB" for idx in range(device_count)} if device_count else None
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            attn_implementation="sdpa" # faster inference
        ).eval()
        print("model loaded", flush=True)
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if hf_device_map is not None:
            print(f"model device map={hf_device_map}", flush=True)
            if all(str(device) == "cpu" for device in hf_device_map.values()):
                raise RuntimeError(
                    "Transformers placed the entire model on CPU despite CUDA being available."
                )
        else:
            first_param_device = next(self.model.parameters()).device
            print(f"model first parameter device={first_param_device}", flush=True)
            if first_param_device.type == "cpu":
                raise RuntimeError(
                    "Transformers placed the model on CPU despite CUDA being available."
                )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
        print("tokenizer created", flush=True)
        
        # decoder-only models need left padding for correct generation order
        self.tokenizer.padding_side = "left"

        # for batching, need to set pad tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task="text-generation",
            temperature=0.1,
            top_p=0.15,
            top_k=0,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            repetition_penalty=1.1,
            do_sample=True,
            batch_size=BATCH_SIZE # for batch support
        )
        eos = self.model.generation_config.eos_token_id
        self.end_ids = (list(eos) if isinstance(eos, (list, tuple)) else [eos]) + [self.tokenizer.eos_token_id]
        print("pipeline created", flush=True)
        
        self.request_queue = asyncio.Queue() # queue for holding requests to process
        self.waiting = []
        self.batch_task = None # current task
        self.batch_lock = asyncio.Lock() # lock for
        self.batch_id = 0
        self.is_processing = False # current state
        print("ready to go", flush=True)
    
    async def start_batch_processor(self):
        """Start the batch processor if it's not already running"""
        async with self.batch_lock:
            if not self.is_processing:
                self.is_processing = True
                self.batch_task = asyncio.create_task(self._batch_processor())
    
    async def _batch_processor(self):
        """Process requests in batches"""
        try:
            while True:
                batch = []
                futures = []
                
                try:
                    if not self.waiting:
                        request, future = await self.request_queue.get()
                        self.waiting.append((request, future, time.time()))
                    batch_start_time = time.time()
                    while len(self.waiting) < BATCH_SIZE and (time.time() - batch_start_time) < BATCH_WAIT_TIME:
                        try:
                            req, fut = await asyncio.wait_for(
                                self.request_queue.get(),
                                timeout=max(0, BATCH_WAIT_TIME - (time.time() - batch_start_time))
                            )
                            self.waiting.append((req, fut, time.time()))
                        except asyncio.TimeoutError:
                            break
                    while not self.request_queue.empty():
                        req, fut = self.request_queue.get_nowait()
                        self.waiting.append((req, fut, time.time()))
                    chosen = pick_batch(self.waiting, time.time(), BATCH_SIZE)
                    picked = set(chosen)
                    batch = [self.waiting[i][0] for i in chosen]
                    futures = [self.waiting[i][1] for i in chosen]
                    self.waiting = [w for i, w in enumerate(self.waiting) if i not in picked]
                    
                    batch_size = len(batch)
                    self.batch_id += 1
                    batch_id = self.batch_id
                    print(
                        f"Processing batch {batch_id} of {batch_size} requests "
                        f"({sum(LONG_PROMPT_MARKER in r['prompt'] for r in batch)} long; "
                        f"waiting after collect: {len(self.waiting) + self.request_queue.qsize()})",
                        flush=True,
                    )
                    
                    prompts = [req["prompt"] for req in batch]
                    
                    max_new_tokens = max(req["max_new_tokens"] for req in batch)
                    
                    # all temps and top_p are same
                    temperature = batch[0]["temperature"] 
                    top_p = batch[0]["top_p"]
                    
                    start_time = time.time()
                    
                    def run(chunk):
                        extra = {}
                        if STOP_AT_FIRST_CODE_BLOCK:
                            extra["stopping_criteria"] = StoppingCriteriaList([FirstCodeBlock(self.tokenizer)])
                        if CODE_BEFORE_END:
                            extra["logits_processor"] = LogitsProcessorList([CodeBeforeEnd(self.tokenizer, self.end_ids)])
                        return self.pipeline(
                            chunk,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            **extra,
                        )

                    try:
                        results = await asyncio.to_thread(run, prompts)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"Batch {batch_id} ran out of GPU memory; retrying its "
                              f"{batch_size} requests one at a time", flush=True)
                        results = []
                        for prompt in prompts:
                            try:
                                results.append((await asyncio.to_thread(run, [prompt]))[0])
                            except torch.cuda.OutOfMemoryError as err:
                                torch.cuda.empty_cache()
                                results.append(err)
                    finally:
                        torch.cuda.empty_cache()

                    response_time = round(time.time() - start_time, 2)
                    print(
                        f"Finished batch {batch_id} in {response_time}s "
                        f"(queued after finish: {len(self.waiting) + self.request_queue.qsize()})",
                        flush=True,
                    )

                    # for every future, set its result
                    for result, future in zip(results, futures):
                        if isinstance(result, Exception):
                            print(f"Error processing request: {result}", flush=True)
                            future.set_exception(result)
                        else:
                            future.set_result({
                                "generated_text": result[0].get("generated_text", str(result)),
                                "response_time_sec": response_time,
                                "batch_size": batch_size
                            })

                        # done with task
                        self.request_queue.task_done()
                
                except Exception as e:
                    print(f"Error processing batch: {str(e)}", flush=True)
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
                    
                    # Mark all tasks as done
                    for _ in range(len(futures)):
                        self.request_queue.task_done()
                
                # Check if queue is empty, if so - sleep briefly to save resources
                if self.request_queue.empty():
                    await asyncio.sleep(0.01)
        
        except asyncio.CancelledError:
            print("Batch processor cancelled", flush=True)
        except Exception as e:
            print(f"Unexpected error in batch processor: {str(e)}", flush=True)
        finally:
            async with self.batch_lock:
                self.is_processing = False
    
    async def generate(self, request_dict):
        """Submit a request to the batch processor"""
        # future is a placeholder for later result
        future = asyncio.Future()
        
        await self.request_queue.put((request_dict, future))
        print(f"Request queued (queue size: {self.request_queue.qsize()})", flush=True)
        
        # start processing batches if not already started
        await self.start_batch_processor()
        
        # wait & return future result
        return await future

@app.post("/generate")
async def generate_text(request: LLMRequest):
    """
    Submits LLMRequest to the local model. If the model has not been intialized before, this function will first have to intialize the model.

    Parameters:
    LLMRequest:
        txt2llm (str): input to llm
        max_new_tokens (int): maximum number of tokens model should generate
        top_p (int): threshold, higher to consider wider range of words
        temperature (int): randomness, higher for more varied outputs

    Returns:
    dict: generated_text (output of LLM) and response_time (time after intializing model until generated text obtained)
    """
    try:
        # Convert request to dict
        request_dict = {
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens,
            "top_p": request.top_p,
            "temperature": request.temperature
        }
        
        # Get the model instance
        model = LLMModel()
        # Submit to the batch processor and wait for result
        start_time = time.time()
        print(f"Request received at {time.strftime('%H:%M:%S', time.localtime(start_time))}", flush=True)
        
        result = await model.generate(request_dict)
        
        print(f"Request completed in {time.time() - start_time:.2f}s", flush=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/critique")
async def critique_image(request: CritiqueRequest):
    if VLM is None:
        raise HTTPException(status_code=503, detail="VLM not loaded; sbatch server.sh with LLMGE_CIRCUIT_VIEW=gemma")
    start_time = time.time()
    try:
        text = await asyncio.to_thread(VLM.critique, request.image_path, request.prompt, request.max_new_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    print(f"Critique done in {time.time() - start_time:.2f}s", flush=True)
    return {"generated_text": text, "response_time_sec": round(time.time() - start_time, 2)}

@app.get("/")
async def root():
    return {"message": "LLM API is running!"}

print('Server running with server-side batching!', flush=True)


@app.on_event("startup")
async def preload_model():
    """Load the model at process start so first request is fast."""
    model = LLMModel()
    await model.start_batch_processor()
    global VLM
    if CIRCUIT_VIEW == "gemma":
        try:
            VLM = await asyncio.to_thread(GemmaCritic)
        except Exception as e:
            print(f"VLM failed to load, /critique disabled: {e}", flush=True)
