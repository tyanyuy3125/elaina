#include "file.h"
#include "config.h"
#include "logger.h"

ELAINA_NAMESPACE_BEGIN

fs::path File::m_current_working_dir{ELAINA_PROJECT_DIR};
fs::path File::m_output_dir{ELAINA_PROJECT_DIR};

fs::path File::cwd() { return m_current_working_dir; }
fs::path File::outputDir() { return m_output_dir; }

fs::path File::projectDir() { return cwd(); }
fs::path File::dataDir() { return fs::weakly_canonical(cwd() / "common"); }
fs::path File::codeDir() { return fs::weakly_canonical(cwd() / "src"); }

fs::path File::resolve(const fs::path &name)
{
	return name.is_absolute() ? name : fs::weakly_canonical(File::cwd() / name);
}

void File::setOutputDir(const fs::path &outputDir)
{
	ELAINA_LOG(Info, "Setting output directory to %s", outputDir.string().c_str());
	if (!fs::exists(outputDir))
	{
		fs::create_directories(outputDir);
	}
	else if (fs::exists(outputDir) && !fs::is_directory(outputDir))
	{
		ELAINA_LOG(Error, "%s is not a directory!", outputDir.string().c_str());
		return;
	}
	m_output_dir = outputDir;
}

void File::setCwd(const fs::path &cwd)
{
	ELAINA_LOG(Info, "Setting working directory to %s", cwd.string().c_str());
	if (!fs::exists(cwd))
	{
		fs::create_directories(cwd);
	}
	else if (fs::exists(cwd) && !fs::is_directory(cwd))
	{
		ELAINA_LOG(Error, "%s is not a directory!", cwd.string().c_str());
		return;
	}
	m_current_working_dir = cwd;
}

json File::loadJSON(const fs::path &filepath)
{
	if (!fs::exists(filepath))
	{
		ELAINA_LOG(Error, "Cannot locate file at %s", filepath.string().c_str());
		return {};
	}
	std::ifstream f(filepath);
	if (f.fail())
	{
		ELAINA_LOG(Error, "Failed to read JSON file at %s", filepath.string().c_str());
		return {};
	}
	json file = json::parse(f, nullptr, true, true);
	return file;
}

void File::saveJSON(const fs::path &filepath, const json &j)
{
	if (!fs::exists(filepath.parent_path()))
		fs::create_directories(filepath.parent_path());
	std::ofstream ofs(filepath);
	ofs << std::setw(4) << j << std::endl;
	ofs.close();
}

ELAINA_NAMESPACE_END