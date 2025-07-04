paddle2onnx --model_dir en_PP-OCRv3_rec_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file model_rec.onnx \
            --enable_dev_version True