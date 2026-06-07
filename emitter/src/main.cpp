// Read the CSV file row by row
// POST each row as JSON to /ingest/telemetry
// Wait N milliseconds between each emission

#include <iostream>   
#include <fstream>    // reading files
#include <sstream>    // parsing strings
#include <string>     
#include <vector>     // vector (like python list)
#include <chrono>     // time
#include <thread>     // sleep

#include <curl/curl.h> // HTTP requests

using namespace std;

//parse a single csv row into fields
vector<string> parseRow (const string& line){
    vector<string> fields;
    stringstream ss(line);
    string field;
    while (getline(ss,field,',')){
        fields.push_back(field);
    }
    return fields;
}


int main (int argc, char* argv[]){

    string filepath = argv[1]; // --file
    int rate_ms = stoi(argv[2]); // --rate in ms
    string chamber = argv[3]; // // --chamber
    
    // opening file
    ifstream file (filepath);
    string line;
    getline (file,line); // reading headers and discarding



    // curl setup
    curl_global_init (CURL_GLOBAL_ALL);
    CURL* curl = curl_easy_init();

    curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8000/ingest/stream");
    curl_easy_setopt (curl, CURLOPT_POST, 1L);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers,"Content-Type: application/json");
    curl_easy_setopt (curl, CURLOPT_HTTPHEADER,headers);
    cout << "--" << headers;
    int row = 0;
    while (getline(file,line)){
        row++;
        if (line.empty()) {
            cout << "Skipping empty row at line: " << row << "\n";
            continue;
        }

        auto fields = parseRow(line);

        if (fields.size() < 3) {
            cout << "Skipping malformed row at line " << row << ": " << line << "\n";
            continue;
    }
        string json = "{"
         "\"chamber_id\":\"" + chamber + "\","
        "\"gas_flow\":" + fields[0] + ","  // build JSON string manually
        "\"temp\":"     + fields[1] + ","  // \" is an escaped quote character
        "\"pressure\":" + fields[2] +
        "}";

        curl_easy_setopt (curl, CURLOPT_POSTFIELDS, json.c_str());
        CURLcode res = curl_easy_perform(curl); 
        if (res != CURLE_OK){
            cerr << "POST failed: " << curl_easy_strerror(res) << "\n"; 
        }
        // else {
        //     cout << "Row " << row << " sent: " << json << "\n";        
        // }



        this_thread::sleep_for(chrono::milliseconds(rate_ms));
    }

    curl_slist_free_all(headers);   // free header memory
    curl_easy_cleanup(curl);        // close curl session
    curl_global_cleanup();          // clean up curl globally
    return 0;                       // exit successfully

}