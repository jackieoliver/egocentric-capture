#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "GPMF_parser.h"
#include "GPMF_utils.h"
#include "GPMF_mp4reader.h"

static void fourcc_to_str(uint32_t fourcc, char out[5]) {
    out[0] = (char)(fourcc & 0xff);
    out[1] = (char)((fourcc >> 8) & 0xff);
    out[2] = (char)((fourcc >> 16) & 0xff);
    out[3] = (char)((fourcc >> 24) & 0xff);
    out[4] = 0;
}

static int parse_keys(const char *csv, uint32_t *out_keys, size_t max_keys, size_t *out_count) {
    *out_count = 0;
    if (!csv || !csv[0]) {
        return 0;
    }
    char *dup = strdup(csv);
    if (!dup) {
        return -1;
    }
    char *saveptr = NULL;
    char *tok = strtok_r(dup, ",", &saveptr);
    while (tok && *out_count < max_keys) {
        while (*tok == ' ' || *tok == '\t') tok++;
        if (strlen(tok) >= 4) {
            out_keys[*out_count] = STR2FOURCC(tok);
            (*out_count)++;
        }
        tok = strtok_r(NULL, ",", &saveptr);
    }
    free(dup);
    return 0;
}

static int key_in_list(uint32_t key, const uint32_t *keys, size_t key_count) {
    if (key_count == 0) return 1;
    for (size_t i = 0; i < key_count; i++) {
        if (keys[i] == key) return 1;
    }
    return 0;
}

static void json_print_string(FILE *out, const char *value) {
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)value; *p; p++) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', out);
            fputc(*p, out);
        } else if (*p == '\n') {
            fputs("\\n", out);
        } else if (*p == '\r') {
            fputs("\\r", out);
        } else if (*p == '\t') {
            fputs("\\t", out);
        } else {
            fputc(*p, out);
        }
    }
    fputc('"', out);
}

static void write_values(FILE *out, const float *values, uint32_t samples, uint32_t elements) {
    fputc('[', out);
    if (samples == 0 || elements == 0) {
        fputc(']', out);
        return;
    }
    if (elements == 1) {
        for (uint32_t i = 0; i < samples; i++) {
            if (i) fputc(',', out);
            fprintf(out, "%.9g", values[i]);
        }
        fputc(']', out);
        return;
    }
    for (uint32_t i = 0; i < samples; i++) {
        if (i) fputc(',', out);
        fputc('[', out);
        for (uint32_t j = 0; j < elements; j++) {
            if (j) fputc(',', out);
            fprintf(out, "%.9g", values[i * elements + j]);
        }
        fputc(']', out);
    }
    fputc(']', out);
}

static void usage(const char *name) {
    fprintf(stderr, "usage: %s --input <mp4> --output <jsonl> [--keys ACCL,GYRO,GPS5,...]\n", name);
}

int main(int argc, char **argv) {
    const char *input_path = NULL;
    const char *output_path = NULL;
    const char *keys_csv = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--input") && i + 1 < argc) {
            input_path = argv[++i];
        } else if (!strcmp(argv[i], "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (!strcmp(argv[i], "--keys") && i + 1 < argc) {
            keys_csv = argv[++i];
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        }
    }

    if (!input_path || !output_path) {
        usage(argv[0]);
        return 2;
    }

    uint32_t keys[64];
    size_t key_count = 0;
    if (parse_keys(keys_csv, keys, 64, &key_count) != 0) {
        fprintf(stderr, "error: keys_parse_failed\n");
        return 2;
    }

    size_t mp4handle = OpenMP4Source((char *)input_path, MOV_GPMF_TRAK_TYPE, MOV_GPMF_TRAK_SUBTYPE, 0);
    if (mp4handle == 0) {
        fprintf(stderr, "error: mp4_open_failed\n");
        return 1;
    }

    FILE *out = fopen(output_path, "w");
    if (!out) {
        CloseSource(mp4handle);
        fprintf(stderr, "error: output_open_failed\n");
        return 1;
    }

    uint32_t payloads = GetNumberPayloads(mp4handle);
    size_t payloadres = 0;
    GPMF_stream ms;
    uint32_t strm_key = STR2FOURCC("STRM");

    for (uint32_t index = 0; index < payloads; index++) {
        double in = 0.0, out_time = 0.0;
        uint32_t payloadsize = GetPayloadSize(mp4handle, index);
        if (payloadsize == 0) continue;
        payloadres = GetPayloadResource(mp4handle, payloadres, payloadsize);
        uint32_t *payload = GetPayload(mp4handle, payloadres, index);
        if (!payload) continue;
        if (GetPayloadTime(mp4handle, index, &in, &out_time) != GPMF_OK) continue;
        if (GPMF_Init(&ms, payload, payloadsize) != GPMF_OK) continue;

        GPMF_ERR ret = GPMF_FindNext(&ms, strm_key, GPMF_RECURSE_LEVELS | GPMF_TOLERANT);
        while (ret == GPMF_OK) {
            if (GPMF_SeekToSamples(&ms) == GPMF_OK) {
                uint32_t key = GPMF_Key(&ms);
                if (key_in_list(key, keys, key_count)) {
                    uint32_t elements = GPMF_ElementsInStruct(&ms);
                    uint32_t samples = GPMF_PayloadSampleCount(&ms);
                    if (samples > 0 && elements > 0) {
                        uint64_t count = (uint64_t)samples * (uint64_t)elements;
                        if (count < 2000000ULL) {
                            uint32_t buf_bytes = (uint32_t)(count * sizeof(float));
                            float *values = (float *)malloc(buf_bytes);
                            if (values) {
                                if (GPMF_OK == GPMF_ScaledData(&ms, values, buf_bytes, 0, samples, GPMF_TYPE_FLOAT)) {
                                    char key_str[5];
                                    fourcc_to_str(key, key_str);
                                    fprintf(out, "{");
                                    fputs("\"payload_index\":", out);
                                    fprintf(out, "%u", index);
                                    fputs(",\"t_in\":", out);
                                    fprintf(out, "%.6f", in);
                                    fputs(",\"t_out\":", out);
                                    fprintf(out, "%.6f", out_time);
                                    fputs(",\"device_id\":", out);
                                    fprintf(out, "%u", GPMF_DeviceID(&ms));
                                    fputs(",\"key\":", out);
                                    json_print_string(out, key_str);
                                    fputs(",\"samples\":", out);
                                    fprintf(out, "%u", samples);
                                    fputs(",\"elements\":", out);
                                    fprintf(out, "%u", elements);
                                    fputs(",\"values\":", out);
                                    write_values(out, values, samples, elements);
                                    fputs("}\n", out);
                                }
                                free(values);
                            }
                        }
                    }
                }
            }
            ret = GPMF_FindNext(&ms, strm_key, GPMF_RECURSE_LEVELS | GPMF_TOLERANT);
        }
    }

    if (payloadres) {
        FreePayloadResource(mp4handle, payloadres);
    }
    fclose(out);
    CloseSource(mp4handle);
    return 0;
}
