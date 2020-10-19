#include <iostream>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<cmath>
#include"graph.h"
#include"block.h"
#include"time.h"
#include<Python.h>
using namespace cv;
using namespace std;
#include <limits>

using namespace cv;

const Scalar BLUE = Scalar(255,0,0); // Background
const Scalar GREEN = Scalar(0,255,0);//Foreground
const Scalar PINK = Scalar(230,130,255); //ProbBackground
const Scalar RED = Scalar(0,0,255);
//int dx[8]={-1,-1,-1,0,0,0,1,1,1},dy[8]={-1,0,1,-1,0,1,-1,0,1};
class GMM
{
public:
    static const int componentsCount = 5;

    GMM( );
    double op( int ci, const Vec3d color ) ;
    int which( const Vec3d color ) ;

    void initLearning();
    void add( int ci, const Vec3d color );
    void endLearning();

    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double coefs[5];
    double mean[5][3];
    double cov[5][3][3];

    double inverseCovs[componentsCount][3][3]; //协方差的逆矩阵
    double covDeterms[componentsCount];  //协方差的行列式

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

Mat global_img,img_circle,rect_img,img,bgcanshu,fgcanshu,End_img;
Mat leftW, upleftW, upW, uprightW;
Mat mask;
GMM bgdGMM, fgdGMM;
Rect rect;
char ch;
bool flag=false;
void showimg();
void onMouse(int event, int x, int y, int flags, void* param);
void showend();
void show_circle();
double energy[100];
int pox;
 double gamma;
double lambda;
double beta ;

GMM::GMM( ){
}
//高斯联合概率
double GMM::op( int ci, const Vec3d color )
{
    double res = 0;
    if( coefs[ci] > 0 )
    {
        Vec3d diff = color;
        diff[0] -= mean[ci][0]; diff[1] -= mean[ci][1]; diff[2] -= mean[ci][2];
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
                   + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
                   + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}

int GMM::which( const Vec3d color )
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = op( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}

void GMM::initLearning()
{
    for( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

void GMM::add( int ci, const Vec3d color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

void GMM::endLearning()
{
    const double variance = 0.01;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
//            cout<<"---"<<endl;
			coefs[ci] = (double)n/totalSampleCount;

            mean[ci][0] = sums[ci][0]/n; mean[ci][1] = sums[ci][1]/n; mean[ci][2] = sums[ci][2]/n;

            cov[ci][0][0] = prods[ci][0][0]/n - mean[ci][0]*mean[ci][0]; cov[ci][0][1] = prods[ci][0][1]/n - mean[ci][0]*mean[ci][1]; cov[ci][0][2] = prods[ci][0][2]/n - mean[ci][0]*mean[ci][2];
            cov[ci][1][0] = prods[ci][1][0]/n - mean[ci][1]*mean[ci][0]; cov[ci][1][1] = prods[ci][1][1]/n - mean[ci][1]*mean[ci][1]; cov[ci][1][2] = prods[ci][1][2]/n - mean[ci][1]*mean[ci][2];
            cov[ci][2][0] = prods[ci][2][0]/n - mean[ci][2]*mean[ci][0]; cov[ci][2][1] = prods[ci][2][1]/n - mean[ci][2]*mean[ci][1]; cov[ci][2][2] = prods[ci][2][2]/n - mean[ci][2]*mean[ci][2];

//			double dtrm = cov[ci][0][0]*(cov[ci][1][1]*cov[ci][2][2]-cov[ci][1][2]*cov[ci][2][1]) - cov[ci][0][1]*(cov[ci][1][0]*cov[ci][2][2]-cov[ci][1][2]*cov[ci][2][0]) +
//                cov[ci][0][2]*(cov[ci][1][0]*cov[ci][2][1]-cov[ci][1][1]*cov[ci][2][0]);
            Mat m = (Mat_<double>(3,3)<<cov[ci][0][0],cov[ci][0][1],cov[ci][0][2],
                                        cov[ci][1][0],cov[ci][1][1],cov[ci][1][2],
                                        cov[ci][2][0],cov[ci][2][1],cov[ci][2][2]);
            double abs = determinant(m);
//            cout<<dtrm<<" "<<abs<<endl;
            if( abs <= std::numeric_limits<double>::epsilon() )
            {
                cov[ci][0][0] += variance;
                cov[ci][1][1] += variance;
                cov[ci][2][2] += variance;
            }
            calcInverseCovAndDeterm(ci);
        }
    }
}
void GMM::calcInverseCovAndDeterm( int ci )
{
    if( coefs[ci] > 0 )
    {
        Mat m = (Mat_<double>(3,3)<<cov[ci][0][0],cov[ci][0][1],cov[ci][0][2],
                                        cov[ci][1][0],cov[ci][1][1],cov[ci][1][2],
                                        cov[ci][2][0],cov[ci][2][1],cov[ci][2][2]);
        covDeterms[ci] =determinant(m);
        Mat inv;
        invert(m,inv);
        inverseCovs[ci][0][0]=inv.at<double>(0,0);
        inverseCovs[ci][0][1]=inv.at<double>(0,1);
        inverseCovs[ci][0][2]=inv.at<double>(0,2);
        inverseCovs[ci][1][0]=inv.at<double>(1,0);
        inverseCovs[ci][1][1]=inv.at<double>(1,1);
        inverseCovs[ci][1][2]=inv.at<double>(1,2);
        inverseCovs[ci][2][0]=inv.at<double>(2,0);
        inverseCovs[ci][2][1]=inv.at<double>(2,1);
        inverseCovs[ci][2][2]=inv.at<double>(2,2);
//        inverseCovs[ci][0][0] =  (cov[ci][1][1]*cov[ci][2][2]-cov[ci][1][2]*cov[ci][2][1])/covDeterms[ci];
//        inverseCovs[ci][1][0] = -(cov[ci][1][0]*cov[ci][2][2]-cov[ci][1][2]*cov[ci][2][0])/covDeterms[ci];
//        inverseCovs[ci][2][0] =  (cov[ci][1][0]*cov[ci][2][1]-cov[ci][1][1]*cov[ci][2][0])/covDeterms[ci];
//        inverseCovs[ci][0][1] = -(cov[ci][0][1]*cov[ci][2][2]-cov[ci][0][2]*cov[ci][2][1])/covDeterms[ci];
//        inverseCovs[ci][1][1] =  (cov[ci][0][0]*cov[ci][2][2]-cov[ci][0][2]*cov[ci][2][0])/covDeterms[ci];
//        inverseCovs[ci][2][1] = -(cov[ci][0][0]*cov[ci][2][1]-cov[ci][0][1]*cov[ci][2][0])/covDeterms[ci];
//        inverseCovs[ci][0][2] =  (cov[ci][0][1]*cov[ci][1][2]-cov[ci][0][2]*cov[ci][1][1])/covDeterms[ci];
//        inverseCovs[ci][1][2] = -(cov[ci][0][0]*cov[ci][1][2]-cov[ci][0][2]*cov[ci][1][0])/covDeterms[ci];
//        inverseCovs[ci][2][2] =  (cov[ci][0][0]*cov[ci][1][1]-cov[ci][0][1]*cov[ci][1][0])/covDeterms[ci];

    }
}

 double calcBeta( )
{
    double beta = 0;
    Vec3d color,diff;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
             color = img.at<Vec3b>(y,x);
            if( x>0 )
            {
                 diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                 diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                 diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                 diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) ); //论文公式（5）

    return beta;
}

 void calcNWeights( )
{
	const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    Vec3d color,diff;
    double temp;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
             color = img.at<Vec3b>(y,x);
            if( x-1>=0 ) // left  //避免图的边界
            {
                 diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                temp=gamma * exp(-beta*diff.dot(diff));
                leftW.at<double>(y,x) =temp;
//                energy[pox]+=temp;

            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                 diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                temp=gammaDivSqrt2 * exp(-beta*diff.dot(diff));
//                energy[pox]+=temp;
                upleftW.at<double>(y,x) = temp;
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                 diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                temp=gamma * exp(-beta*diff.dot(diff));
//                energy[pox]+=temp;
                upW.at<double>(y,x) =temp;
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                 diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                temp=gammaDivSqrt2 * exp(-beta*diff.dot(diff));
//                energy[pox]+=temp;
                uprightW.at<double>(y,x) = temp;
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
//    cout<<"aaaaaaaaaaaaaaaaa"<<endl;
//    cout<<energy[pox]<<endl;
}

void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo( GC_BGD );

    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, imgSize.width-rect.x);
    rect.height = min(rect.height, imgSize.height-rect.y);

    mask(rect).setTo( Scalar(GC_PR_FGD) );
}

 void initGMMs(  )
{
    gamma=50,lambda=9*gamma;
    const int kMeansItCount = 1;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    vector<Vec3f> bgdSamples, fgdSamples;
    int x,y;
    for( x = 0;x< img.rows; x++ )
    {
        for(y = 0; y< img.cols;y++ )
        {
			if( mask.at<uchar>(x,y) == GC_BGD || mask.at<uchar>(x,y) == GC_PR_BGD )
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(x,y) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(x,y) );
        }
    }
    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

	bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.add( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.add( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();
}

void Assign()
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    int x,y;
    for(x=0; x< img.rows; x++ )
    {
        for( y = 0; y< img.cols; y++ )
        {
            Vec3d color = img.at<Vec3b>(x,y);
            if(mask.at<uchar>(x,y) == GC_BGD || mask.at<uchar>(x,y) == GC_PR_BGD){
                bgdGMM.add(bgdGMM.which(color),color);
            }
            else{
                fgdGMM.add(fgdGMM.which(color),color);
            }
        }
    }
}


 void learnGMMs()
{
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}


 void build(Graph<double,double,double>& graph )
{
    int x,y;
    double a,b;
    Vec3b color;
    double w;
    double fromSource, toSink;
    for( x= 0; x < img.rows; x++ )
    {
        for( y = 0; y < img.cols; y++)
        {
            int numid = graph.add_node();
            color = img.at<Vec3b>(x,y);
            if( mask.at<uchar>(x,y) == GC_PR_BGD || mask.at<uchar>(x,y) == GC_PR_FGD )
            {

//                a=0,b=0;
//                for(int i=0;i<5;i++){
//                    a+=bgdGMM.coefs[i]*bgdGMM.op(i,color);
//                    b+=fgdGMM.coefs[i]*fgdGMM.op(i,color);
//                }
                int ida=bgdGMM.which(color);
                int idb=fgdGMM.which(color);
				fromSource = -log( bgdGMM.op(ida,color) );
                toSink = -log( fgdGMM.op(idb,color) );
            }
            else if( mask.at<uchar>(x,y) == GC_BGD )
            {
				fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.add_tweights( numid, fromSource, toSink );
			if(y>0 )
            {
                w = leftW.at<double>(x,y-1);
                graph.add_edge( numid, numid-1, w, w );
            }
            if( x>0 && y>0 )
            {
                 w = upleftW.at<double>(x,y);
                graph.add_edge( numid, numid-img.cols-1, w, w );
            }
            if( x>0 )
            {
                 w = upW.at<double>(x-1,y);
                graph.add_edge( numid, numid-img.cols, w, w );
            }
            if( y<img.cols-1 && x>0 )
            {
                 w = uprightW.at<double>(x-1,y+1);
                graph.add_edge( numid, numid-img.cols+1, w, w );
            }
        }
    }
}

void towfrom( Graph<double,double,double>& graph )
{
    int x,y;
    for(x = 0; x < mask.rows;x++ )
    {
        for( y = 0; y < mask.cols; y++ )
        {
			if( mask.at<uchar>(x,y) == GC_PR_BGD || mask.at<uchar>(x,y) == GC_PR_FGD )
            {
                if( graph.what_segment( x*mask.cols+y ) )
                    mask.at<uchar>(x,y) = GC_PR_BGD;
                else
                    mask.at<uchar>(x,y) = GC_PR_FGD;
            }
        }
    }
}
bool check(int a,int b){
    if(a==GC_BGD||a==GC_PR_BGD){
        if(b==GC_FGD||b==GC_PR_FGD)
            return true;
        return false;
    }
    else{
        if(b==GC_BGD||b==GC_PR_BGD)
            return true;
        return false;
    }

}
void caltemp(){
    Vec3b color;
    int type;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            color=img.at<Vec3b>(i,j);
            type=mask.at<uchar>(i,j);
            if(type==GC_BGD||type==GC_PR_BGD){
//                for(int i=0;i<5;i++)
                    int id=bgdGMM.which(color);
                    energy[pox]+=(-log(bgdGMM.op(id,color)*bgdGMM.coefs[id]));
//
            }
            else{
                int id=fgdGMM.which(color);
                energy[pox]+=(-log(fgdGMM.op(id,color)*fgdGMM.coefs[id]));
            }
            if(i>0&&check(type,mask.at<uchar>(i-1,j))){
                energy[pox]+=upW.at<double>(i,j);
//                cout<<upW.at<double>(i,j)<<endl;
            }
            if(i>0&&j>0&&check(type,mask.at<uchar>(i-1,j-1))){
                energy[pox]+=upleftW.at<double>(i,j);
            }
            if(j>0&&check(type,mask.at<uchar>(i,j-1))){
                energy[pox]+=leftW.at<double>(i,j);
            }
            if(i>0&&j<img.cols-1&&check(type,mask.at<uchar>(i-1,j+1))){
                energy[pox]+=uprightW.at<double>(i,j);
            }
        }
    }
    cout<<energy[pox]<<endl;
}
void grabCut(Mat  &img, Mat &mask, Rect rect,
                  int iterCount, bool flag )
{
//                    cout<<"dao"<<endl;
//    Mat img = _img.getMat();

//    GMM bgdGMM, fgdGMM;
    int vernum=img.rows*img.cols;
    int edgenum= 2 * (4 * vernum - 3 * (img.cols + img.rows) + 2);

        if( flag ){
            initMaskWithRect( mask, img.size(), rect );
            initGMMs( );
            beta = calcBeta(  );
            calcNWeights( );
        }
//        cout<<"cao"<<endl;
    for( int i = 0; i < iterCount; i++ )
    {
        Graph<double,double,double> graph(vernum,edgenum);
        Assign();
        learnGMMs( );
        build( graph );
        graph.maxflow();
        towfrom( graph);
//        caltemp();
//        pox++;
    }
//    caltemp();
//    pox++;
}
int main(){
//    clock_t s=clock();
//    int a=0;
//    for(int i=0;i<100000000;i++){
//        a++;
//    }
//    clock_t e=clock();
//    cout<<double(e-s)/1000;

//    Py_Initialize();
//	Py_Initialize(); /*初始化python解释器,告诉编译器要用的python编译器*/
//	PyRun_SimpleString("print('Hello')"); /*调用python文件*/
//	Py_Finalize(); /*结束python解释器，释放资源*/
//	system("pause");

    clock_t Start,End;
    global_img = cv::imread("3.jpg");
//    cv::resize(global_img,global_img,Size(400,600));
    img=global_img.clone();
    img_circle=img.clone();
    imshow("scr",global_img);

    setMouseCallback("scr",onMouse,0);
    waitKey(0);


    bool ff=true;
    bool first=true;
    while(ff){
        cout<<"--"<<endl;
        ch=(char)cvWaitKey(0);
        cout<<ch<<endl;
//        ff=false;
        switch(ch){
            case 'b':
                break;
            case 'f':
                break;
            case 'c'://不传入任何信息继续迭代
                    Start=clock();
                    grabCut(img,mask,rect,3,first);

                    End=clock();
                    cout<<"运行时间"<<double(End-Start)/1000<<endl;
                    first=false;
                showend();
                cout<<"ok"<<endl;
                break;
            case 'e':
                ff=false;
                break;
            default:
                break;
        }
    }


//    cout<<"---"<<endl;
//    con.iter(itercount);
//    cout<<"1"<<endl;
    cout<<"img "<<img.rows<<" "<<img.cols<<endl;
    cout<<"mask "<<mask.rows<<" "<<mask.cols<<endl;
}
void showend(){
    End_img=img.clone();
    Vec3b v(255,255,255);
    for(int i=0;i<End_img.rows;i++){
        for(int j=0;j<End_img.cols;j++){
            if((mask.at<uchar>(i,j)==GC_PR_BGD||mask.at<uchar>(i,j)==GC_BGD)){
                End_img.at<Vec3b>(i,j)[0]=255;
                End_img.at<Vec3b>(i,j)[1]=255;
                End_img.at<Vec3b>(i,j)[2]=255;
            }
        }
    }
    imshow("End",End_img);
}
void showimg(){
    rect_img=img.clone();
    rectangle(rect_img,rect,BLUE);
    imshow("rect_img",rect_img);
//    waitKey(0);
}
void show_circle(){
    imshow("Circle",img_circle);
}
void onMouse(int event, int x, int y, int flags, void* param){
    switch(event){
    case EVENT_LBUTTONDOWN:
        if(!flag){
            rect.x=x;
            rect.y=y;
            rect.height=1;
            rect.width=1;
        }
        else{
            if(ch=='b'){
                mask.at<uchar>(y,x)=GC_BGD;
                circle(img_circle,Point(x,y),1,PINK);
            }
            else if(ch=='f'){
                mask.at<uchar>(y,x)=GC_FGD;
                circle(img_circle,Point(x,y),1,BLUE);
            }
            show_circle();
        }
        break;
    case EVENT_MOUSEMOVE:
        if(flags&EVENT_FLAG_LBUTTON){
            if(!flag){
                rect.width=x-rect.x;
                rect.height=y-rect.y;
//                cv::rectangle(global_img,rect,cv::Scalar(0,0,255));
                showimg();
            }
            else{
                if(ch=='b'){
                    mask.at<uchar>(y,x)=GC_BGD;
                    circle(img_circle,Point(x,y),1,PINK);
                }
                else if(ch=='f'){
                    mask.at<uchar>(y,x)=GC_FGD;
                    circle(img_circle,Point(x,y),1,BLUE);
                }
                show_circle();
            }
        }

        break;
    case EVENT_LBUTTONUP:
        if(!flag){
            rect.width=x-rect.x;
            rect.height=y-rect.y;
            showimg();
            flag=true;
        }
        else{

            if(ch=='b'){
                mask.at<uchar>(y,x)=GC_BGD;
                circle(img_circle,Point(y,x),1,PINK);
            }
            else if(ch=='f'){
                mask.at<uchar>(y,x)=GC_FGD;
                circle(img_circle,Point(y,x),1,BLUE);
            }
            show_circle();
        }
    }
}
