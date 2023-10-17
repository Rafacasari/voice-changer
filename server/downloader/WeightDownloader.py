import os
from concurrent.futures import ThreadPoolExecutor

from downloader.Downloader import download
from mods.log_control import VoiceChangaerLogger
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from Exceptions import WeightDownladException

logger = VoiceChangaerLogger.get_instance().getLogger()

downloadQueueSize = 0
downloadParams = []
   
def downloadWeight(voiceChangerParams: VoiceChangerParams):
    content_vec_500_onnx = voiceChangerParams.content_vec_500_onnx
    hubert_base = voiceChangerParams.hubert_base
    hubert_base_jp = voiceChangerParams.hubert_base_jp
    hubert_soft = voiceChangerParams.hubert_soft
    nsf_hifigan = voiceChangerParams.nsf_hifigan
    crepe_onnx_full = voiceChangerParams.crepe_onnx_full
    crepe_onnx_tiny = voiceChangerParams.crepe_onnx_tiny
    
    rmvpe = voiceChangerParams.rmvpe
    rmvpe_onnx = voiceChangerParams.rmvpe_onnx
    rmvpe_only = voiceChangerParams.rmvpe_only
    
    weight_files = [content_vec_500_onnx, hubert_base, hubert_base_jp, hubert_soft,
                    nsf_hifigan, crepe_onnx_full, crepe_onnx_tiny, rmvpe]

    # file exists check (currently only for rvc)

    def addToQueue(url, saveTo):
        global downloadQueueSize
        global downloadParams
        print(downloadQueueSize)
        
        downloadParams.append({
            "url": url,
            "saveTo": saveTo,
            "position": downloadQueueSize
        })
        downloadQueueSize += 1
    
    if os.path.exists(hubert_base) is False and not rmvpe_only:
        addToQueue("https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt", hubert_base)
        
    if os.path.exists(hubert_base_jp) is False and not rmvpe_only:
        addToQueue("https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt", hubert_base_jp)
        
    if os.path.exists(hubert_soft) is False and not rmvpe_only:
        addToQueue("https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/embedder/hubert-soft-0d54a1f4.pt", hubert_soft)
        
    if os.path.exists(nsf_hifigan) is False and not rmvpe_only:
        addToQueue("https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_20221211/model.bin", nsf_hifigan)
        
    nsf_hifigan_config = os.path.join(os.path.dirname(nsf_hifigan), "config.json")
    if os.path.exists(nsf_hifigan_config) is False and not rmvpe_only:
        addToQueue("https://huggingface.co/wok000/weights/raw/main/ddsp-svc30/nsf_hifigan_20221211/config.json", nsf_hifigan_config)
        
    nsf_hifigan_onnx = os.path.join(os.path.dirname(nsf_hifigan), "nsf_hifigan.onnx")
    if os.path.exists(nsf_hifigan_onnx) is False and not rmvpe_only:
        addToQueue("https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_onnx_20221211/nsf_hifigan.onnx", nsf_hifigan_onnx)

    if os.path.exists(crepe_onnx_full) is False and not rmvpe_only:
        addToQueue("https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/full.onnx", crepe_onnx_full)
        
    if os.path.exists(crepe_onnx_tiny) is False and not rmvpe_only:
        addToQueue("https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/tiny.onnx", crepe_onnx_tiny)

    if os.path.exists(content_vec_500_onnx) is False and not rmvpe_only:
        addToQueue("https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx", content_vec_500_onnx)
        
    if os.path.exists(rmvpe) is False:
        addToQueue("https://huggingface.co/wok000/weights/resolve/main/rmvpe/rmvpe_20231006.pt", rmvpe)
        
    if os.path.exists(rmvpe_onnx) is False:
        addToQueue("https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx", rmvpe_onnx)

    with ThreadPoolExecutor() as pool:
        pool.map(download, downloadParams)

    # unnecessary for now
    # if os.path.exists(hubert_base) is False or os.path.exists(hubert_base_jp) is False or os.path.exists(hubert_soft) is False or os.path.exists(nsf_hifigan) is False or os.path.exists(nsf_hifigan_config) is False:
        # raise WeightDownladException()

    # ファイルサイズをログに書き込む。（デバッグ用）
    for weight in weight_files:
        if os.path.exists(weight):
            file_size = os.path.getsize(weight)
            logger.debug(f"weight file [{weight}]: {file_size}")
        else:
            logger.warning(f"weight file is missing. {weight}")
            #raise WeightDownladException()
