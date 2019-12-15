cmake_minimum_required(VERSION 3.0.2)

project(test)

install(
    DIRECTORY ispd2005 DESTINATION ${PROJECT_NAME}
    )

install(
    DIRECTORY ispd2015 DESTINATION ${PROJECT_NAME}
    )
