#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
using namespace std;

const int R = 2000;     //����Ԫ��, ��R����ͬ��Ƶ��, ��ֵԽ��,��ӦƵ��ԽС,������0
const double A = 1.25;  //�������A>1�ĸ�����, ��������С��1��,�ƺ�Ҳ����
const double C = 1.0;   //���C�ǲ���Ҫ��,һ��ȡ1, ���Կ�����������з��ӷ�ĸ����Լ�����C
double pf[R];           //ֵΪ0~1֮��, �ǵ���f(r)���ۼ�ֵ

void generate()
{
	double sum = 0.0;
	for (int i = 0; i < R; i++)    
	{        
		sum += C/pow((double)(i+2), A);  //λ��Ϊi��Ƶ��,һ����r��(����), �ۼ����   
	}
	for (int i = 0; i < R; i++)    
	{        
		if (i == 0)            
			pf[i] = C/pow((double)(i+2), A)/sum;        
		else            
			pf[i] = pf[i-1] + C/pow((double)(i+2), A)/sum;    
	}
}

void pick(int n)
{
	srand(time(00));
	//����n����
	for (int i = 0; i < n; i++)    
	{
		int index = 0;
		double data = (double)rand()/RAND_MAX;  //����һ��0~1����
		while (data > pf[index])   //������,ֱ���ҵ�һ������С��ֵ,��ô��Ӧ��index�����������
			index++;
		//cout<<index<<"\t";
		cout<<pf[index]<<"\n";
	}
}

int main()
{    
	generate();    
	pick(1000);       
	return 1;
}