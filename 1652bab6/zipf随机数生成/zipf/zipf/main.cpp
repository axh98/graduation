#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
using namespace std;

const int R = 2000;     //数据元素, 有R个不同的频率, 数值越大,对应频率越小,逐渐趋于0
const double A = 1.25;  //定义参数A>1的浮点数, 后来测试小于1的,似乎也可以
const double C = 1.0;   //这个C是不重要的,一般取1, 可以看到下面计算中分子分母可以约掉这个C
double pf[R];           //值为0~1之间, 是单个f(r)的累加值

void generate()
{
	double sum = 0.0;
	for (int i = 0; i < R; i++)    
	{        
		sum += C/pow((double)(i+2), A);  //位置为i的频率,一共有r个(即秩), 累加求和   
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
	//产生n个数
	for (int i = 0; i < n; i++)    
	{
		int index = 0;
		double data = (double)rand()/RAND_MAX;  //生成一个0~1的数
		while (data > pf[index])   //找索引,直到找到一个比他小的值,那么对应的index就是随机数了
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