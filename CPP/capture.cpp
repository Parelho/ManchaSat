#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

int main() {
    while (true) {
        cv::VideoCapture cap("libcamerasrc ! video/x-raw,width=480,height=270,format=BGR ! videoconvert ! appsink", cv::CAP_GSTREAMER);
        cv::Mat frame; 

        if (!cap.isOpened()) {
            std::cerr << "Erro ao abrir a câmera" << std::endl;
            return -1;
        }

        std::cout << "Câmera aberta com sucesso" << std::endl;

        // Define as propriedades da câmera
        // cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
        // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 256);

        // Verificar as propriedades da câmera
        // std::cout << "Frame Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
        // std::cout << "Frame Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
        // cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
        // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

        // Capturar uma imagem
        bool isSuccess = cap.read(frame);

        if (!isSuccess) {
            std::cerr << "Erro ao capturar a imagem" << std::endl;
            return -1;
        } else {
            std::cout << "Imagem capturada com sucesso" << std::endl;
        }

        if (frame.empty()) {
            std::cerr << "Erro: a imagem está vazia" << std::endl;
            return -1;
        }

        isSuccess = cv::imwrite("../image.jpg", frame);
        if (!isSuccess) {
            std::cerr << "Erro ao salvar a imagem" << std::endl;
        } else {
            std::cout << "Imagem capturada e salva com sucesso" << std::endl;
        }

        cap.release();
    }

    return 0;
}