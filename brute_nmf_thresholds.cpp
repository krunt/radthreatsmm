#include <bits/stdc++.h>
#include <sys/time.h>

using namespace std;

typedef long long ll;
typedef pair<int, int> ppi;

ll getMs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (ll)tv.tv_sec * 1000LL + tv.tv_usec / 1000LL;
}

double rand01() { return (double)rand() / RAND_MAX; }

static vector<string> strSplit(const string &s, char sep = ' ')
{
    vector<string> ret;
    int from = 0;
    while (1) {
      int nxt = s.find(sep, from);
      if (nxt == string::npos) {
        ret.push_back(s.substr(from));
        break;
      }
      ret.push_back(s.substr(from, nxt - from));
      from = nxt + 1;
    }
    return ret;
}

static std::string strJoin(const std::vector<std::string> &vec, char sep)
{
    if (vec.empty()) return "";
    std::string ret = vec[0];
    for (int i = 1; i < vec.size(); ++i) {
      ret += sep;
      ret += vec[i];
    }
    return ret;
}

const int NUM_CLASSES = 6;
const int MAX_ID = 10000;
const float BG_THRESHOLD = 10.0;
float g_export_snr[MAX_ID * NUM_CLASSES];
float g_export_time[MAX_ID * NUM_CLASSES];
float g_export_sresbgs[MAX_ID * NUM_CLASSES];

int g_truth_source_id[MAX_ID];
float g_truth_source_time[MAX_ID];
float g_truth_speed_offset[MAX_ID];

int g_runid_cnt = 0;
int g_runid_map[MAX_ID];

const int NUM_VARS = 6;
float g_coeff_range[NUM_VARS][2] = {
    { 1.0, 1.9 },
    { 1.0, 1.9 },
    { 1.0, 1.9 },
    { 1.0, 1.9 },
    { 1.0, 1.9 },
    { 1.0, 3.0 },
    /*
    { 8, 12 },
    { 8, 12 },
    { 8, 12 },
    { 8, 12 },
    { 8, 12 },
    { 8, 16 },
    */

    /*
    { 10, 11 },
    { 10, 11 },
    { 10, 11 },
    { 10, 11 },
    { 10, 11 },
    { 10, 11 },
    */
};
//int g_coeff_nsteps[NUM_VARS] = { 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 6 };
int g_coeff_nsteps[NUM_VARS] = { 9, 9, 9, 9, 9, 14, }; // 2, 2, 2, 2, 2, 2 };

const float S_FN = -2;
const float S_FP = -2;
const float S_TN = 6;
const float S_type = 1;

vector<float> g_coeff_vec[NUM_VARS];

int get_coeff_nsteps_max()
{
    int ret = 1;
    for (int i = 0; i < NUM_VARS; ++i) {
        ret *= g_coeff_nsteps[i];
    }
    return ret;
}

void decode_coeff(int code, vector<int> &outcode) 
{
    outcode.resize(NUM_VARS);
    for (int i = 0; i < NUM_VARS; ++i) {
        int nstep = g_coeff_nsteps[i];
        outcode[i] = code % nstep;
        code /= nstep;
    }
}

vector<float> incode2thresh(const vector<int> &incode)
{
    vector<float> othresh(NUM_VARS);
    for (int i = 0; i < NUM_VARS; ++i) {
        othresh[i] = g_coeff_vec[i][incode[i]];
    }
    return othresh;
}

float compute_score(const vector<int> &incode)
{
    auto thresh = incode2thresh(incode);
    float score = 0;
    for (int id = 0; id < g_runid_cnt; ++id) {
        int best_cid = -1;
        for (int cid = 0; cid < NUM_CLASSES; ++cid) {
            if (g_export_sresbgs[id * NUM_CLASSES + cid] < BG_THRESHOLD) { //thresh[cid + NUM_CLASSES]) {
                continue;
            }
            if (g_export_snr[id * NUM_CLASSES + cid] < thresh[cid]) {
                continue;
            }
            if (best_cid == -1 
                    || g_export_snr[id * NUM_CLASSES + best_cid] < g_export_snr[id * NUM_CLASSES + cid])
            {
                best_cid = cid;
            }
        }

        if (g_truth_source_id[id] != 0) {
            if (best_cid == -1) {
                score += S_FN;
            } else {
                float distance_in_meters = abs(g_export_time[id * NUM_CLASSES + best_cid] 
                        - g_truth_source_time[id]);
                distance_in_meters *= g_truth_speed_offset[id];
                if (distance_in_meters < 1) {
                    float distance_bonus = cos(distance_in_meters * M_PI/2);
                    score += distance_bonus;

                    if (g_truth_source_id[id] == best_cid) {
                        score += S_type;
                    }
                } else {
                    score += S_FP;
                }
            }
        } else {
            if (best_cid == -1) {
                score += S_TN;
            } else {
                score += S_FP;
            }
        }
    }
    return score;
}

int main()
{
    memset(g_runid_map, -1, sizeof(g_runid_map));

    for (int i = 0; i < NUM_VARS; ++i) {
        float from = g_coeff_range[i][0];
        float to = g_coeff_range[i][1];
        int num_steps = g_coeff_nsteps[i];
        float step = (to - from) / (num_steps-1);
        for (int j = 0; j < num_steps; ++j) {
            g_coeff_vec[i].push_back(from + j * step);
        }
    }

    {
        string line;
        ifstream fd("export.csv");
        getline(fd, line);
        for (;getline(fd, line);) {
            int runid, source_id;
            float snr, ti, toffs, sresbgs;
            sscanf(line.c_str(), "%d,%f,%f,%d,%f,%f", &runid, &snr, &ti, &source_id, &toffs, &sresbgs);
            assert(runid >= 100000 && runid < 110000);
            runid -= 100000;
            if (g_runid_map[runid] == -1) {
                g_runid_map[runid] = g_runid_cnt++;
            }
            int tid = g_runid_map[runid];
            g_export_snr[NUM_CLASSES * tid + source_id - 1] = snr;
            g_export_time[NUM_CLASSES * tid + source_id - 1] = ti + toffs;
            g_export_sresbgs[NUM_CLASSES * tid + source_id - 1] = sresbgs;
        }
    }

    {
        string line;
        ifstream fd("answerKey.csv");
        getline(fd, line);
        for (;getline(fd, line);) {
            int runid, source_id, part;
            float source_time, speed_offset;
            sscanf(line.c_str(), "%d,%d,%f,%d,%f", &runid, &source_id, &source_time, &part, &speed_offset);
            assert(runid >= 100000 && runid < 110000);
            runid -= 100000;
            if (g_runid_map[runid] == -1) {
                continue;
            }
            int idx = g_runid_map[runid];
            g_truth_source_id[idx] = source_id;
            g_truth_source_time[idx] = source_time;
            g_truth_speed_offset[idx] = speed_offset;
        }
    }

    float best_score = 0;
    int best_code = -1;
    int maxcode = get_coeff_nsteps_max();
    vector<int> coeff(6);
    cout << "maxcode = " << maxcode << endl;
    for (int code = 0; code < maxcode; ++code) {
        if (code % 100000 == 0) {
            cout << code / 100000 << endl;
        }
        decode_coeff(code,  coeff);
        float score = compute_score(coeff);
        if (best_score < score) {
            best_code = code;
            best_score = score;
        }
    }

    decode_coeff(best_code, coeff);
    auto bc = incode2thresh(coeff);
    //cout << best_code << " " << bc[0] << " " << bc[1] << " " << bc[2] << " " << 
        //bc[3] << " " << bc[4] << " " << bc[5] << endl;
    for (int i = 0; i < NUM_VARS; ++i) {
        cout << i << ": " << bc[i] << endl;
    }

    return 0;
}
