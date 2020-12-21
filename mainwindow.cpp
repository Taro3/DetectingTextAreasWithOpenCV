#include <QGraphicsScene>

#include <tesseract/baseapi.h>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , _net()
{
    ui->setupUi(this);
}

MainWindow::~MainWindow() {
    delete ui;
}

cv::Mat MainWindow::detectTextAreas(QImage &image, std::vector<cv::Rect> &areas) {
    float confThreshold = 0.5;
    float nmsThreshold = 0.4;
    int inputWidth = 320;
    int inputHeight = 320;
    std::string model = "./frozen_east_text_detection.pb";
    // Load DNN network.
    if (_net.empty()) {
        _net = cv::dnn::readNet(model);
    }

    std::vector<cv::Mat> outs;
    std::vector<std::string> layerNames(2);
    layerNames[0] = "feature_fusion/Conv_7/Sigmoid";
    layerNames[1] = "feature_fusion/concat_3";

    cv::Mat frame = cv::Mat(image.height(), image.width(), CV_8UC3, image.bits(), image.bytesPerLine()).clone();
    cv::Mat blob;

    cv::dnn::blobFromImage(frame, blob, 1.0, cv::Size(inputWidth, inputHeight), cv::Scalar(123.68, 116.78, 103.94),
                           true, false);
    _net.setInput(blob);
    _net.forward(outs, layerNames);

    cv::Mat scores = outs[0];
    cv::Mat geometry = outs[1];

    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    decode(scores, geometry, confThreshold, boxes, confidences);

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    cv::Point2f ratio((float)frame.cols / inputWidth, (float)frame.rows / inputHeight);
    cv::Scalar green = cv::Scalar(0, 255, 0);

    for (size_t i = 0; i < indices.size(); ++i) {
        cv::RotatedRect& box = boxes[indices[i]];
        cv::Rect area = box.boundingRect();
        area.x *= ratio.x;
        area.width *= ratio.x;
        area.y *= ratio.y;
        area.height *= ratio.y;
        areas.push_back(area);
        cv::rectangle(frame, area, green, 1);
        QString index = QString("%1").arg(i);
        cv::putText(frame, index.toStdString(), cv::Point2f(area.x, area.y - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, green,
                    1);
    }

    return frame;
}

void MainWindow::on_pushButton_clicked() {
    QImage image;
    image.load("testimage.jpg");
    image = image.convertToFormat(QImage::Format_RGB888);

    auto tessApi = tesseract::TessBaseAPI();
    tessApi.Init(TESSDATA_PREFIX, "eng");
    tessApi.SetImage(image.bits(), image.width(), image.height(), 3, image.bytesPerLine());

    std::vector<cv::Rect> areas;
    cv::Mat newImage = detectTextAreas(image, areas);
    showImage(newImage);
    ui->plainTextEdit->setPlainText("");
    for (cv::Rect &rect : areas) {
        tessApi.SetRectangle(rect.x, rect.y, rect.width, rect.height);
        char* outText = tessApi.GetUTF8Text();
        ui->plainTextEdit->setPlainText(ui->plainTextEdit->toPlainText() + outText);
        delete [] outText;
    }

    tessApi.End();
}

void MainWindow::showImage(cv::Mat mat) {
    QImage image(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);

    QPixmap pixmap = QPixmap::fromImage(image);
    QGraphicsScene* imageScene = new QGraphicsScene(this);
    ui->graphicsView->setScene(imageScene);
    imageScene->clear();
    ui->graphicsView->resetTransform();
    auto currentImage = imageScene->addPixmap(pixmap);
    imageScene->update();
    ui->graphicsView->setSceneRect(pixmap.rect());
}

void MainWindow::decode(const cv::Mat &scores, const cv::Mat &geometry, float scoreThresh,
                        std::vector<cv::RotatedRect> &detections, std::vector<float> &confidences)
{
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4);
    CV_Assert(scores.size[0] == 1); CV_Assert(scores.size[1] == 1);
    CV_Assert(geometry.size[0] == 1);  CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]);
    CV_Assert(scores.size[3] == geometry.size[3]);

    detections.clear();
    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y) {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x) {
            float score = scoresData[x];
            if (score < scoreThresh)
                continue;

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
            cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
            cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}
