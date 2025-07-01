#include "exec.h"
#include "util/check.h"
#include <iostream>
#include <string>
#include <variant>
#include <set>
#include <magic_enum/magic_enum.hpp>

#include "core/device/buffer.h"

#include "core/device/cuda.h"
#include "util/tonemapping.cuh"
#include "integrator/guided/integrator.h"
#include "integrator/uniform/integrator.h"

#include "core/problem.h"

#include "util/network.h"

#include <filesystem>

#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/io/IO.h>

using namespace elaina;

std::string get_current_time()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm *timeinfo = std::localtime(&t);
    char buffer[20];
    std::strftime(buffer, 20, "%Y-%m-%d %H:%M:%S", timeinfo);
    return std::string(buffer);
}

void run_expr(fs::path conf_path)
{
    if (!fs::exists(conf_path))
    {
        ELAINA_LOG(Error, "Configuration file does not exist: %s", conf_path.c_str());
        return;
    }

    std::ifstream conf_file(conf_path);
    if (!conf_file.is_open())
    {
        ELAINA_LOG(Error, "Configuration file does not exist: %s", conf_path.c_str());
        return;
    }

    nlohmann::json conf_json;
    nlohmann::json result_json;
    try
    {
        conf_file >> conf_json;
    }
    catch (nlohmann::json::parse_error &e)
    {
        ELAINA_LOG(Error, "Failed to parse JSON: %s", e.what());
        return;
    }
    auto dimensionality = json_get_or_throw<int>(conf_json, "dimensionality");
    fs::path basePath = json_get_or_throw<string>(conf_json, "base_path");

    std::string expName = json_get_or_throw<string>(conf_json, "exp_name");
    std::ofstream confFileCopy(basePath / expName / "conf.json");
    ELAINA_LOG(Success, "Configuration file copied to %s", (basePath / expName / "conf.json").c_str());

    confFileCopy << conf_json.dump(4) << std::endl;
    auto scene_section = json_get_or_throw<json>(conf_json, "scene");
    auto integrator_section = json_get_or_throw<json>(conf_json, "integrator");
    auto integrator_type = json_get_or_throw<string>(integrator_section, "type");
    auto integrator_setting = json_get_or_throw<json>(integrator_section, "setting");
    std::variant<std::monostate, UniformIntegrator<2>, GuidedIntegrator<2>, UniformIntegrator<3>, GuidedIntegrator<3>> integrator = {};
    std::variant<std::monostate, Problem<2>, Problem<3>> scene;
    if (dimensionality == 2)
    {
        using Problem = Problem<2>;
        using UniformIntegrator = UniformIntegrator<2>;
        using GuidedIntegrator = GuidedIntegrator<2>;
        scene = Problem();
        std::get<Problem>(scene).loadConfig(scene_section);
        if (integrator_type == "uniform")
        {
            integrator.emplace<UniformIntegrator>(std::get<Problem>(scene), integrator_setting, basePath / expName);
        }
        else if (integrator_type == "guided")
        {
            auto network_section = json_get_or_throw<json>(conf_json, "network");
            integrator.emplace<GuidedIntegrator>(std::get<Problem>(scene), integrator_setting, basePath / expName);
            std::get<GuidedIntegrator>(integrator).resetNetwork(network_section);
        }
        else
        {
            ELAINA_LOG(Error, "Unrecognized integrator type.");
            exit(1);
        }
    }
    else if (dimensionality == 3)
    {
        using Problem = Problem<3>;
        using UniformIntegrator = UniformIntegrator<3>;
        using GuidedIntegrator = GuidedIntegrator<3>;
        scene = Problem();
        std::get<Problem>(scene).loadConfig(scene_section);
        if (integrator_type == "uniform")
        {
            integrator.emplace<UniformIntegrator>(std::get<Problem>(scene), integrator_setting, basePath / expName);
        }
        else if (integrator_type == "guided")
        {
            auto network_section = json_get_or_throw<json>(conf_json, "network");
            integrator.emplace<GuidedIntegrator>(std::get<Problem>(scene), integrator_setting, basePath / expName);
            std::get<GuidedIntegrator>(integrator).resetNetwork(network_section);
        }
        else
        {
            ELAINA_LOG(Error, "Unrecognized integrator type.");
            exit(1);
        }
    }
    else
    {
        ELAINA_LOG(Error, "Unsupported dimensionality.");
        exit(1);
    }

    auto export_section = json_get_or_throw<json>(conf_json, "export");
    auto integrator_channels = json_get_or_throw<json>(integrator_section, "channels");
    std::set<ExportImageChannel> integrator_channels_set;
    for (const auto &integrator_channel_string : integrator_channels)
    {
        auto integrator_channel = magic_enum::enum_cast<ExportImageChannel>(integrator_channel_string.get<string>());
        if (!integrator_channel.has_value())
        {
            ELAINA_LOG(Error, "Unrecognized integrator channel, skipping...");
        }
        integrator_channels_set.insert(integrator_channel.value());
    }
    auto print_network = json_get_optional<bool>(conf_json, "print_network", false);

    std::visit([&](auto &&obj)
               {
        if constexpr (std::is_same_v<std::decay_t<decltype(obj)>, std::monostate>) {
            ELAINA_SHOULDNT_GO_HERE;
        } else {
            using VectorType = typename std::decay_t<decltype(obj)>::VectorType;
            for (const auto &integrator_channel : integrator_channels_set)
            {
                switch(integrator_channel)
                {
                    case ExportImageChannel::SOLUTION:
                    {
                        uint64_t duration = obj.solve();
                        result_json["duration"] = duration;
                        break;
                    }
                    case ExportImageChannel::DIRICHLET_SDF:
                        obj.renderDirichletSDF();
                        break;
                    case ExportImageChannel::NEUMANN_SDF:
                        obj.renderSilhouetteSDF();
                        break;
                    case ExportImageChannel::SOURCE:
                        obj.renderSource();
                        break;
                    default:
                        ELAINA_SHOULDNT_GO_HERE;
                        break;
                }
            }
            if (print_network)
            {
                if (dimensionality == 3)
                {
                    std::cout << "great!" << std::endl;
                    obj.queryNetwork(Vector3f(0.0f, -0.21f, 0.0f));
                }
                else
                {
                    obj.queryNetwork(VectorType::Zero());
                }
            }
            for (const auto &export_method : export_section)
            {
                auto export_type = json_get_or_throw<string>(export_method, "type");
                auto export_channel_string = json_get_or_throw<string>(export_method, "channel");
                auto export_channel = magic_enum::enum_cast<ExportImageChannel>(export_channel_string);
                auto export_file_name = json_get_or_throw<string>(export_method, "file_name");
                if (!export_channel.has_value())
                {
                    ELAINA_LOG(Error, "Unrecognized export channel, skipping...");
                }
                if (export_type == "image")
                {
                    obj.exportImage(export_channel.value(), export_file_name);
                }
                else if (export_type == "energy")
                {
                    auto export_tone_string = json_get_or_throw<string>(export_method, "tone");
                    auto export_tone = magic_enum::enum_cast<ToneMapping>(export_tone_string);
                    if(export_tone.has_value())
                    {
                        obj.exportEnergy(export_channel.value(), export_tone.value(), export_file_name);
                    }
                    else
                    {
                        ELAINA_LOG(Error, "Unrecognized tone mapping method, skipping...");
                    }
                }
            }
        } }, integrator);
    
    result_json["timestamp"] = get_current_time();
    std::ofstream resultFile(basePath / expName / "result.json");
    resultFile << result_json.dump(4) << std::endl;
    ELAINA_LOG(Success, "Result file written to %s", (basePath / expName / "result.json").c_str());
}
