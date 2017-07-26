#include <cstdio>
#include <iostream>
#include <sstream>
#include<vector>
using namespace std;

vector<pair<string, vector<int> > > lines_;
int main(){
	string str = "dds df	nmi			1 10\n  ddas  22";
	stringstream ss(str);
	string tmp;
	while(ss >> tmp){
		cout << tmp << "$\n";
	}

	string filename = "good";
	vector<int> labels;
	labels.push_back(1);
	labels.push_back(3);
	lines_.push_back(make_pair(filename, labels));
	for(size_t i=0; i < lines_.size(); i++){
		cout << lines_[i].first << endl;
		for (size_t j = 0; j < lines_[i].second.size(); j++){
			cout << lines_[i].second[j] << ' ';
		}
		cout << '$' << endl;
	}
	vector<int> label_shape;
	label_shape.push_back(32);
	label_shape.push_back(381);
	for (size_t i = 0; i < label_shape.size(); i++){
		cout << label_shape[i] << ' ';
	}
	cout << '$' << endl;

	return 0;
}
