/* 
* Copyright (C) 2020-2026 MEmilio
*
* Authors: Martin Siggel, Daniel Abele, Martin J. Kuehn, Jan Kleinert, Khoa Nguyen
*
* Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

//Includes from pymio
#include "abm/person_id.h"
#include "pybind_util.h"
#include "utils/custom_index_array.h"
#include "utils/parameter_set.h"
#include "utils/index.h"

//Includes from MEmilio
#include "abm/simulation.h"

#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/operators.h"
#include <cstdint>
#include <type_traits>

#include "boost/filesystem.hpp"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include <boost/algorithm/string.hpp>

namespace py = pybind11;

struct LogNewInfectionsAndShedding : mio::LogAlways { //AS
    using Type = std::tuple<
        int,
        double>; //First tuple entry is number of new infections per time point, second tuple entry ist summed exposure rates (global_parameters.get<InfectionRateFromViralShed>()[{virus}] * infection.get_infectivity(t + dt / 2)) per time point
    static Type log(const mio::abm::Simulation<>& sim)
    {
        int new_infections = 0;
        double shedding    = 0.0;
        
        for (auto&& person : sim.get_model().get_persons()) {
            auto time_since_transmission =  sim.get_time() - person.get_infection().get_start_date(); //before: get_time_since_transmission()
            if (time_since_transmission.hours() >= 0 && time_since_transmission.hours() < 1) {
                new_infections += 1;
            }
            if (person.is_infected(sim.get_time())) {
                auto& infection = person.get_infection();
                auto virus      = infection.get_virus_variant();
                for (mio::abm::CellIndex cell : person.get_cells()) {
                    shedding += sim.get_model().parameters.get<mio::abm::InfectionRateFromViralShed>()[{virus}] *
                                infection.get_viral_shed(sim.get_time() + mio::abm::hours(1) / 2); //before: get_infectivity()
                }
            }
        }
        return std::make_tuple(new_infections, shedding);
    }
};

//AgentId logger
struct LogAgentIds : mio::LogOnce { //AS
    using Type = std::vector<mio::abm::PersonId>;
    static Type log(const mio::abm::Simulation<>& sim)
    {
        std::vector<mio::abm::PersonId> agent_ids{};
        for (auto&& person : sim.get_model().get_persons()) {
            agent_ids.push_back(person.get_id());
        }
        return agent_ids;
    }
};

//time point logger
struct LogTimePoint : mio::LogAlways { //AS
    using Type = double;
    static Type log(const mio::abm::Simulation<>& sim)
    {
        return sim.get_time().hours();
    }
};

//LocationId logger
struct LogLocationIds : mio::LogOnce {//AS
    using Type = std::vector<std::tuple<mio::abm::LocationId, mio::abm::LocationType>>;
    static Type log(const mio::abm::Simulation<>& sim)
    {
        std::vector<std::tuple<mio::abm::LocationId, mio::abm::LocationType>> location_ids{};
        for (auto&& location : sim.get_model().get_locations()) {
            location_ids.push_back(std::make_tuple(location.get_id(), location.get_type()));
        }
        return location_ids;
    }
};

//agent logger
struct LogPersonsPerLocationAndInfectionTime : mio::LogAlways { //AS
    using Type = std::vector<std::tuple<mio::abm::LocationId, mio::abm::LocationType, mio::abm::PersonId,
                                        mio::abm::TimeSpan, mio::abm::InfectionState>>;//, int>>;
    static Type log(const mio::abm::Simulation<>& sim)
    {
        std::vector<std::tuple<mio::abm::LocationId, mio::abm::LocationType, mio::abm::PersonId, mio::abm::TimeSpan,
                               mio::abm::InfectionState>>//, int>>
            location_ids_person{};
        for (auto&& person : sim.get_model().get_persons()) {
            //int ww_id = sim.get_model().get_location(person.get_location()); //before: .get_wastewater_id();
            auto time_since_transmission =  sim.get_time() - person.get_infection().get_start_date(); //before: get_time_since_transmission()
            location_ids_person.push_back(std::make_tuple(person.get_location(), person.get_location_type(),
                                                          person.get_id(), time_since_transmission, //before: .get_time_since_transmission(),
                                                          person.get_infection_state(sim.get_time())));//, ww_id));
        }
        return location_ids_person;
    }
};

// ###################################################################################################################################################################

void write_mapping_to_file(std::string filename, std::map<int, std::vector<std::string>>& mapping)
{
    auto file = fopen(filename.c_str(), "w");
    if (file == NULL) {
        mio::log(mio::LogLevel::warn, "Could not open file {}", filename);
    }
    else {
        for (auto it = mapping.begin(); it != mapping.end(); it++) {
            fprintf(file, "%d", it->first);
            for (auto s : it->second) {
                fprintf(file, " %s", s.c_str());
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }
}

mio::AgeGroup determine_age_group(uint32_t age)
{
    if (age <= 4) {
        return mio::AgeGroup(0);
    }
    else if (age <= 15) {
        return mio::AgeGroup(1);
    }
    else if (age <= 34) {
        return mio::AgeGroup(2);
    }
    else if (age <= 59) {
        return mio::AgeGroup(3);
    }
    else if (age <= 79) {
        return mio::AgeGroup(4);
    }
    else if (age > 79) {
        return mio::AgeGroup(5);
    }
    else {
        return mio::AgeGroup(0);
    }
}



void initialize_model(mio::abm::Model& model, std::string person_file, std::string hosp_file, std::string outfile,
                      size_t max_work_size, size_t max_school_size)
{
    // Mapping of ABM locations to traffic areas/cells
    // - each traffic area is mapped to a vector containing strings with LocationType and LocationId
    std::map<int, std::vector<std::string>> loc_area_mapping;
    // Mapping of traffic data location ids to ABM location ids
    std::map<int, mio::abm::LocationId> home_locations;
    std::map<int, mio::abm::LocationId> shop_locations;
    std::map<int, mio::abm::LocationId> event_locations;
    std::map<int, mio::abm::LocationId> school_locations;
    std::map<int, mio::abm::LocationId> work_locations;
    std::vector<mio::abm::LocationId> hospitals;
    std::vector<mio::abm::LocationId> icus;
    std::map<std::pair<mio::abm::LocationType, mio::abm::LocationId>, mio::abm::LocationId> hosp_to_icu;
    std::vector<double> hospital_weights;
    std::vector<double> icu_weights;
    // Mapping of assigned agents to school and work locations
    std::map<int, std::map<mio::abm::LocationId, size_t>> school_sizes;
    std::map<int, std::map<mio::abm::LocationId, size_t>> work_sizes;

    // Read in hospitals
    const boost::filesystem::path h = hosp_file;
    if (!boost::filesystem::exists(h)) {
        mio::log_error("Cannot read in data. Hospital file does not exist.");
    }
    // File pointer
    std::fstream fin_hosp;
    // Open an existing file
    fin_hosp.open(hosp_file, std::ios::in);
    std::vector<int32_t> row_hosp;
    std::vector<std::string> row_string_hosp;
    std::string line_hosp;
    // Read the Titles from the Data file
    std::getline(fin_hosp, line_hosp);
    line_hosp.erase(std::remove(line_hosp.begin(), line_hosp.end(), '\r'), line_hosp.end());
    std::vector<std::string> titles_hosp;
    boost::split(titles_hosp, line_hosp, boost::is_any_of(","));
    uint32_t count_of_titles_hosp              = 0;
    std::map<std::string, uint32_t> index_hosp = {};
    for (auto const& title : titles_hosp) {
        index_hosp.insert({title, count_of_titles_hosp});
        row_string_hosp.push_back(title);
        count_of_titles_hosp++;
    }
    while (std::getline(fin_hosp, line_hosp)) {
        row_hosp.clear();

        // read columns in this row
        split_line(line_hosp, &row_hosp); 
        line_hosp.erase(std::remove(line_hosp.begin(), line_hosp.end(), '\r'), line_hosp.end());

        int beds          = row_hosp[index_hosp["beds"]];
        int icu_beds      = row_hosp[index_hosp["icu_beds"]];
        int hospital_zone = row_hosp[index_hosp["hospital_zone"]];
        auto hosp         = model.add_location(mio::abm::LocationType::Hospital);
        hospitals.push_back(hosp);
        hospital_weights.push_back(beds);
        std::string loc =
            "0" + std::to_string(static_cast<int>(mio::abm::LocationType::Hospital)) + std::to_string(hosp.get());
        auto zone_iter = loc_area_mapping.find(hospital_zone);
        if (zone_iter == loc_area_mapping.end()) {
            loc_area_mapping.insert({hospital_zone, {loc}});
        }
        else {
            loc_area_mapping[hospital_zone].push_back(loc);
        }
        // Add icu if there is one
        if (icu_beds > 0) {
            auto icu = model.add_location(mio::abm::LocationType::ICU);
            icus.push_back(icu);
            icu_weights.push_back(icu_beds);
            hosp_to_icu.insert({std::make_pair(mio::abm::LocationType::Hospital, hosp), icu});
            std::string loc_icu =
                "0" + std::to_string(static_cast<int>(mio::abm::LocationType::ICU)) + std::to_string(icu.get());
            zone_iter = loc_area_mapping.find(hospital_zone);
            if (zone_iter == loc_area_mapping.end()) {
                loc_area_mapping.insert({hospital_zone, {loc_icu}});
            }
            else {
                loc_area_mapping[hospital_zone].push_back(loc_icu);
            }
        }
    }

    // Read in persons
    const boost::filesystem::path p = person_file;
    if (!boost::filesystem::exists(p)) {
        mio::log_error("Cannot read in data. File does not exist.");
    }
    // File pointer
    std::fstream fin;

    // Open an existing file
    fin.open(person_file, std::ios::in);
    std::vector<int32_t> row;
    std::vector<std::string> row_string;
    std::string line;

    // Read the Titles from the Data file
    std::getline(fin, line);
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
    std::vector<std::string> titles;
    boost::split(titles, line, boost::is_any_of(","));
    uint32_t count_of_titles              = 0;
    std::map<std::string, uint32_t> index = {};
    for (auto const& title : titles) {
        index.insert({title, count_of_titles});
        row_string.push_back(title);
        count_of_titles++;
    }

    while (std::getline(fin, line)) {
        row.clear();

        // read columns in this row
        split_line(line, &row);
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

        uint32_t age = row[index["age"]];

        int home_id   = row[index["home_id"]];
        int home_zone = row[index["home_zone"]];

        mio::abm::LocationId home;

        auto iter_home = home_locations.find(home_id);
        // check whether home location already exists in model
        if (iter_home == home_locations.end()) {
            // if home location does not exists yet, create new location and insert it to mapping
            home = model.add_location(mio::abm::LocationType::Home);
            home_locations.insert({home_id, home});
            std::string loc =
                "0" + std::to_string(static_cast<int>(mio::abm::LocationType::Home)) + std::to_string(home.get());
            auto zone_iter = loc_area_mapping.find(home_zone);
            if (zone_iter == loc_area_mapping.end()) {
                loc_area_mapping.insert({home_zone, {loc}});
            }
            else {
                loc_area_mapping[home_zone].push_back(loc);
            }
        }
        else {
            home = home_locations[home_id];
        }
        // Add person to model and assign home location to it
        auto pid     = model.add_person(home, determine_age_group(age));
        auto& person = model.get_person(pid);
        person.set_assigned_location(mio::abm::LocationType::Home, home, model.get_id());
        model.get_location(home).increase_size();

        int shop_id   = row[index["shop_id"]];
        int shop_zone = row[index["shop_zone"]];

        mio::abm::LocationId shop;

        auto iter_shop = shop_locations.find(shop_id);
        // Check whether shop location already exists in model
        if (iter_shop == shop_locations.end()) {
            // Create shop location and add it to mapping
            shop = model.add_location(mio::abm::LocationType::BasicsShop);
            // Shops with ids -1 are individual locations each
            if (shop_id != -1) {
                shop_locations.insert({shop_id, shop});
            }
            std::string loc =
                "0" + std::to_string(static_cast<int>(mio::abm::LocationType::BasicsShop)) + std::to_string(shop.get());
            auto zone_iter = loc_area_mapping.find(shop_zone);
            if (zone_iter == loc_area_mapping.end()) {
                loc_area_mapping.insert({shop_zone, {loc}});
            }
            else {
                loc_area_mapping[shop_zone].push_back(loc);
            }
        }
        else {
            shop = shop_locations[shop_id];
        }
        // Assign shop to person
        person.set_assigned_location(mio::abm::LocationType::BasicsShop, shop, model.get_id());
        model.get_location(shop).increase_size();

        int event_id   = row[index["event_id"]];
        int event_zone = row[index["event_zone"]];

        mio::abm::LocationId event;

        auto iter_event = event_locations.find(event_id);
        // Check whether event location already exists in model
        if (iter_event == event_locations.end()) {
            //Create event location and add it to mapping
            event = model.add_location(mio::abm::LocationType::SocialEvent);
            // Events with id -1 are individual locations each
            if (event_id != -1) {
                event_locations.insert({event_id, event});
            }
            std::string loc = "0" + std::to_string(static_cast<int>(mio::abm::LocationType::SocialEvent)) +
                              std::to_string(event.get());
            auto zone_iter = loc_area_mapping.find(event_zone);
            if (zone_iter == loc_area_mapping.end()) {
                loc_area_mapping.insert({event_zone, {loc}});
            }
            else {
                loc_area_mapping[event_zone].push_back(loc);
            }
        }
        else {
            event = event_locations[event_id];
        }
        // Assign event location to person
        person.set_assigned_location(mio::abm::LocationType::SocialEvent, event, model.get_id());
        model.get_location(event).increase_size();

        // Check if person is school-aged
        if (person.get_age() == mio::AgeGroup(1)) {
            int school_id   = row[index["school_id"]];
            int school_zone = row[index["school_zone"]];

            mio::abm::LocationId school;

            auto iter_school = school_locations.find(school_id);
            // Check whether school location is already in model
            if (iter_school == school_locations.end()) {
                // Add schools locations to model and insert it in mapping
                school = model.add_location(mio::abm::LocationType::School);
                // schools with id -1 are individual locations each
                if (school_id != -1) {
                    school_locations.insert({school_id, school});
                    // Add school to size map
                    school_sizes[school_id].insert({school, 1});
                }
                std::string loc = "0" + std::to_string(static_cast<int>(mio::abm::LocationType::School)) +
                                  std::to_string(school.get());
                auto zone_iter = loc_area_mapping.find(school_zone);
                if (zone_iter == loc_area_mapping.end()) {
                    loc_area_mapping.insert({school_zone, {loc}});
                }
                else {
                    loc_area_mapping[school_zone].push_back(loc);
                }
            }
            else {
                school = school_locations[school_id];
                if (school_sizes[school_id][school] == max_school_size) {
                    // Check if a new school has to be open or if there still is a school that has capacity
                    bool found = false;
                    for (auto const& [key, val] : school_sizes[school_id]) {
                        if (val < max_school_size) {
                            found = true;
                            school_sizes[school_id][key] += 1;
                            school = key;
                            break;
                        }
                    }
                    if (!found) {
                        // Create new school
                        school = model.add_location(mio::abm::LocationType::School);
                        school_locations.insert({school_id, school});
                        // Add school to size map
                        school_sizes[school_id].insert({school, 1});
                        std::string loc = "0" + std::to_string(static_cast<int>(mio::abm::LocationType::School)) +
                                          std::to_string(school.get());
                        auto zone_iter = loc_area_mapping.find(school_zone);
                        if (zone_iter == loc_area_mapping.end()) {
                            loc_area_mapping.insert({school_zone, {loc}});
                        }
                        else {
                            loc_area_mapping[school_zone].push_back(loc);
                        }
                    }
                }
                else {
                    school_sizes[school_id][school] += 1;
                }
            }
            // Assign school location to person
            person.set_assigned_location(mio::abm::LocationType::School, school, model.get_id());
            model.get_location(school).increase_size();
        }
        // Check if person is work-aged
        if (person.get_age() == mio::AgeGroup(2) || person.get_age() == mio::AgeGroup(3)) {
            int work_id   = row[index["work_id"]];
            int work_zone = row[index["work_zone"]];

            if (work_zone == -2) {
                mio::log_error("Person with id {} has work age but no work zone", row[index["puid"]]);
            }

            mio::abm::LocationId work;

            auto iter_work = work_locations.find(work_id);
            // Check whether work location already exists in model
            if (iter_work == work_locations.end()) {
                // Add work location to model and insert it in mapping
                work = model.add_location(mio::abm::LocationType::Work);
                // Locations with id -1 are individual locations each
                if (work_id != -1) {
                    work_locations.insert({work_id, work});
                    // Add work to size map
                    work_sizes[work_id].insert({work, 1});
                }
                std::string loc =
                    "0" + std::to_string(static_cast<int>(mio::abm::LocationType::Work)) + std::to_string(work.get());
                auto zone_iter = loc_area_mapping.find(work_zone);
                if (zone_iter == loc_area_mapping.end()) {
                    loc_area_mapping.insert({work_zone, {loc}});
                }
                else {
                    loc_area_mapping[work_zone].push_back(loc);
                }
            }
            else {
                work = work_locations[work_id];
                if (work_sizes[work_id][work] == max_work_size) {
                    // Check if a new work has to be opened or if there still is a school that has capacity
                    bool found = false;
                    for (auto const& [key, val] : work_sizes[work_id]) {
                        if (val < max_work_size) {
                            found = true;
                            work_sizes[work_id][key] += 1;
                            work = key;
                            break;
                        }
                    }
                    if (!found) {
                        // Create new work
                        work = model.add_location(mio::abm::LocationType::Work);
                        work_locations.insert({work_id, work});
                        // Add work to size map
                        work_sizes[work_id].insert({work, 1});
                        std::string loc = "0" + std::to_string(static_cast<int>(mio::abm::LocationType::Work)) +
                                          std::to_string(work.get());
                        auto zone_iter = loc_area_mapping.find(work_zone);
                        if (zone_iter == loc_area_mapping.end()) {
                            loc_area_mapping.insert({work_zone, {loc}});
                        }
                        else {
                            loc_area_mapping[work_zone].push_back(loc);
                        }
                    }
                }
                else {
                    work_sizes[work_id][work] += 1;
                }
            }
            // Assign work location to person
            person.set_assigned_location(mio::abm::LocationType::Work, work, model.get_id());
            model.get_location(work).increase_size();
        }
        // Assign Hospital and ICU
        size_t hosp = mio::DiscreteDistribution<size_t>::get_instance()(model.get_rng(), hospital_weights);
        person.set_assigned_location(mio::abm::LocationType::Hospital, hospitals[hosp], model.get_id());
        model.get_location(hospitals[hosp]).increase_size();
        if (hosp_to_icu.count(std::make_pair(mio::abm::LocationType::Hospital, hospitals[hosp])) > 0) {
            person.set_assigned_location(
                mio::abm::LocationType::ICU,
                hosp_to_icu[std::make_pair(mio::abm::LocationType::Hospital, hospitals[hosp])], model.get_id());
            model.get_location(hosp_to_icu[std::make_pair(mio::abm::LocationType::Hospital, hospitals[hosp])])
                .increase_size();
        }
        else {
            size_t icu = mio::DiscreteDistribution<size_t>::get_instance()(model.get_rng(), icu_weights);
            person.set_assigned_location(mio::abm::LocationType::ICU, icus[icu], model.get_id());
            model.get_location(icus[icu]).increase_size();
        }
    }

    write_mapping_to_file(outfile, loc_area_mapping);
}



PYBIND11_MODULE(_simulation_abm, m)
{
    pymio::iterable_enum<mio::abm::InfectionState>(m, "InfectionState")
        .value("Susceptible", mio::abm::InfectionState::Susceptible)
        .value("Exposed", mio::abm::InfectionState::Exposed)
        .value("InfectedNoSymptoms", mio::abm::InfectionState::InfectedNoSymptoms)
        .value("InfectedSymptoms", mio::abm::InfectionState::InfectedSymptoms)
        .value("InfectedSevere", mio::abm::InfectionState::InfectedSevere)
        .value("InfectedCritical", mio::abm::InfectionState::InfectedCritical)
        .value("Recovered", mio::abm::InfectionState::Recovered)
        .value("Dead", mio::abm::InfectionState::Dead);

    pymio::iterable_enum<mio::abm::ProtectionType>(m, "ProtectionType")
        .value("NoProtection", mio::abm::ProtectionType::NoProtection)
        .value("NaturalInfection", mio::abm::ProtectionType::NaturalInfection)
        .value("GenericVaccine", mio::abm::ProtectionType::GenericVaccine);

    pymio::iterable_enum<mio::abm::VirusVariant>(m, "VirusVariant").value("Wildtype", mio::abm::VirusVariant::Wildtype);

    pymio::iterable_enum<mio::abm::LocationType>(m, "LocationType")
        .value("Home", mio::abm::LocationType::Home)
        .value("School", mio::abm::LocationType::School)
        .value("Work", mio::abm::LocationType::Work)
        .value("SocialEvent", mio::abm::LocationType::SocialEvent)
        .value("BasicsShop", mio::abm::LocationType::BasicsShop)
        .value("Hospital", mio::abm::LocationType::Hospital)
        .value("ICU", mio::abm::LocationType::ICU)
        .value("Car", mio::abm::LocationType::Car)
        .value("PublicTransport", mio::abm::LocationType::PublicTransport)
        .value("TransportWithoutContact", mio::abm::LocationType::TransportWithoutContact);

    pymio::iterable_enum<mio::abm::TestType>(m, "TestType")
        .value("Generic", mio::abm::TestType::Generic)
        .value("Antigen", mio::abm::TestType::Antigen)
        .value("PCR", mio::abm::TestType::PCR);

    pymio::bind_class<mio::abm::TimeSpan, pymio::EnablePickling::Never>(m, "TimeSpan")
        .def(py::init<int>(), py::arg("seconds") = 0)
        .def_property_readonly("seconds", &mio::abm::TimeSpan::seconds)
        .def_property_readonly("hours", &mio::abm::TimeSpan::hours)
        .def_property_readonly("days", &mio::abm::TimeSpan::days)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * int{})
        .def(py::self *= int{})
        .def(py::self / int{})
        .def(py::self /= int{})
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self <= py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self);

    m.def("seconds", &mio::abm::seconds);
    m.def("minutes", &mio::abm::minutes);
    m.def("hours", &mio::abm::hours);
    m.def("days", py::overload_cast<int>(&mio::abm::days));

    pymio::bind_class<mio::abm::TimePoint, pymio::EnablePickling::Never>(m, "TimePoint")
        .def(py::init<int>(), py::arg("seconds") = 0)
        .def_property_readonly("seconds", &mio::abm::TimePoint::seconds)
        .def_property_readonly("days", &mio::abm::TimePoint::days)
        .def_property_readonly("hours", &mio::abm::TimePoint::hours)
        .def_property_readonly("day_of_week", &mio::abm::TimePoint::day_of_week)
        .def_property_readonly("hour_of_day", &mio::abm::TimePoint::hour_of_day)
        .def_property_readonly("time_since_midnight", &mio::abm::TimePoint::time_since_midnight)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self <= py::self)
        .def(py::self > py::self)
        .def(py::self >= py::self)
        .def(py::self - py::self)
        .def(py::self + mio::abm::TimeSpan{})
        .def(py::self += mio::abm::TimeSpan{})
        .def(py::self - mio::abm::TimeSpan{})
        .def(py::self -= mio::abm::TimeSpan{});

    pymio::bind_class<mio::abm::TestParameters, pymio::EnablePickling::Never>(m, "TestParameters")
        .def(py::init<double, double, mio::abm::TimeSpan, mio::abm::TestType>())
        .def_readwrite("sensitivity", &mio::abm::TestParameters::sensitivity)
        .def_readwrite("specificity", &mio::abm::TestParameters::specificity)
        .def_readwrite("required_time", &mio::abm::TestParameters::required_time)
        .def_readwrite("type", &mio::abm::TestParameters::type);

    pymio::bind_CustomIndexArray<mio::UncertainValue<double>, mio::abm::VirusVariant, mio::AgeGroup>(
        m, "_AgeParameterArray");
    pymio::bind_CustomIndexArray<mio::abm::TestParameters, mio::abm::TestType>(m, "_TestData");
    pymio::bind_Index<mio::abm::ProtectionType>(m, "ProtectionTypeIndex");
    pymio::bind_ParameterSet<mio::abm::ParametersBase, pymio::EnablePickling::Never>(m, "ParametersBase");
    pymio::bind_class<mio::abm::Parameters, pymio::EnablePickling::Never, mio::abm::ParametersBase>(m, "Parameters")
        .def(py::init<int>())
        .def("check_constraints", &mio::abm::Parameters::check_constraints);

    pymio::bind_ParameterSet<mio::abm::LocalInfectionParameters, pymio::EnablePickling::Never>(
        m, "LocalInfectionParameters")
        .def(py::init<size_t>());

    pymio::bind_class<mio::abm::LocationId, pymio::EnablePickling::Never>(m, "LocationId")
        .def(py::init<uint32_t>(), py::arg("id"))
        .def("index", &mio::abm::LocationId::get);

    pymio::bind_class<mio::abm::PersonId, pymio::EnablePickling::Never>(m, "PersonId")
        .def(py::init<uint64_t>(), py::arg("id"))
        .def("index", &mio::abm::PersonId::get);

    pymio::bind_class<mio::abm::Person, pymio::EnablePickling::Never>(m, "Person")
        .def("set_assigned_location", py::overload_cast<mio::abm::LocationType, mio::abm::LocationId, int>(
                                          &mio::abm::Person::set_assigned_location))
        .def_property_readonly("location", py::overload_cast<>(&mio::abm::Person::get_location, py::const_))
        .def_property_readonly("age", &mio::abm::Person::get_age)
        .def_property_readonly("is_in_quarantine", &mio::abm::Person::is_in_quarantine);

    pymio::bind_class<mio::abm::TestingCriteria, pymio::EnablePickling::Never>(m, "TestingCriteria")
        .def(py::init<const std::vector<mio::AgeGroup>&, const std::vector<mio::abm::InfectionState>&>(),
             py::arg("age_groups"), py::arg("infection_states"));

    pymio::bind_class<mio::abm::TestingScheme, pymio::EnablePickling::Never>(m, "TestingScheme")
        .def(py::init<const mio::abm::TestingCriteria&, mio::abm::TimeSpan, mio::abm::TimePoint, mio::abm::TimePoint,
                      const mio::abm::TestParameters&, double>(),
             py::arg("testing_criteria"), py::arg("testing_validity_period"), py::arg("start_date"),
             py::arg("end_date"), py::arg("test_parameters"), py::arg("probability"));

    pymio::bind_class<mio::abm::ProtectionEvent, pymio::EnablePickling::Never>(m, "ProtectionEvent")
        .def(py::init<mio::abm::ProtectionType, mio::abm::TimePoint>(), py::arg("type"), py::arg("time"))
        .def_readwrite("type", &mio::abm::ProtectionEvent::type)
        .def_readwrite("time", &mio::abm::ProtectionEvent::time);

    pymio::bind_class<mio::abm::TestingStrategy, pymio::EnablePickling::Never>(m, "TestingStrategy")
        .def(py::init<const std::vector<mio::abm::TestingStrategy::LocalStrategy>&,
                      const std::vector<mio::abm::TestingStrategy::LocalStrategy>&>());

    pymio::bind_class<mio::abm::Infection, pymio::EnablePickling::Never>(m, "Infection")
        .def(py::init([](mio::abm::Model& model, mio::abm::Person& person, mio::abm::VirusVariant variant,
                         mio::abm::TimePoint start_date, mio::abm::InfectionState start_state, bool detected //bool shift_init, double shift_rate
                         ) {
            auto rng = mio::abm::PersonalRandomNumberGenerator(model.get_rng(), person);
            //auto time_since_infection =  mio::abm::Simulation::get_time() - person.get_infection().get_start_date(); //before: get_time_since_transmission()
            return mio::abm::Infection(rng, variant, person.get_age(), model.parameters, start_date, start_state,
                                       person.get_latest_protection(start_date), detected); //before: person.get_latest_protection()
        }))
        .def("get_infection_start", [](const mio::abm::Infection& infection) {
            return infection.get_start_date();
        });
        //.def("get_infection_start", &mio::abm::Infection::get_start_date()) //get_infection_start
        //.def("get_time_in_state", [](mio::abm::Infection& self, mio::abm::InfectionState state) {
            //return self.get_time_in_state(state);
        //});

    pymio::bind_class<mio::abm::Location, pymio::EnablePickling::Never>(m, "Location")
        .def_property_readonly("type", &mio::abm::Location::get_type)
        .def_property_readonly("id", &mio::abm::Location::get_id)
        .def_property("infection_parameters",
                      py::overload_cast<>(&mio::abm::Location::get_infection_parameters, py::const_),
                      [](mio::abm::Location& self, mio::abm::LocalInfectionParameters params) {
                          self.get_infection_parameters() = params;
                      });

    //copying and moving of ranges enabled below, see PYMIO_IGNORE_VALUE_TYPE
    pymio::bind_Range<decltype(std::declval<const mio::abm::Model>().get_locations())>(m, "_ModelLocationsRange");
    pymio::bind_Range<decltype(std::declval<const mio::abm::Model>().get_persons())>(m, "_ModelPersonsRange");

    pymio::bind_class<mio::abm::Trip, pymio::EnablePickling::Never>(m, "Trip")
        .def(py::init<mio::abm::PersonId, mio::abm::TimePoint, mio::abm::LocationId>(), py::arg("person_id"),
             py::arg("time"), py::arg("destination"))
        .def_readwrite("person_id", &mio::abm::Trip::person_id)
        .def_readwrite("trip_time", &mio::abm::Trip::trip_time)
        .def_readwrite("destination", &mio::abm::Trip::destination);

    pymio::bind_class<mio::abm::TripList, pymio::EnablePickling::Never>(m, "TripList")
        .def(py::init<>())
        .def("add_trips", &mio::abm::TripList::add_trips, py::arg("trips") = std::vector<mio::abm::Trip>())
        .def("next_trip", &mio::abm::TripList::get_next_trip)
        .def("num_trips", &mio::abm::TripList::num_trips);

    pymio::bind_class<mio::abm::Model, pymio::EnablePickling::Never>(m, "Model")
        .def(py::init<int32_t>())
        .def("add_location", &mio::abm::Model::add_location, py::arg("location_type"), py::arg("num_cells") = 1)
        .def("add_person", py::overload_cast<mio::abm::LocationId, mio::AgeGroup>(&mio::abm::Model::add_person),
             py::arg("location_id"), py::arg("age_group"))
        .def("assign_location",
             py::overload_cast<mio::abm::PersonId, mio::abm::LocationId>(&mio::abm::Model::assign_location),
             py::arg("person_id"), py::arg("location_id"))
        .def_property_readonly("locations", py::overload_cast<>(&mio::abm::Model::get_locations, py::const_),
                               py::keep_alive<1, 0>{}) //keep this model alive while contents are referenced in ranges
        .def_property_readonly("persons", py::overload_cast<>(&mio::abm::Model::get_persons, py::const_),
                               py::keep_alive<1, 0>{})
        .def_property(
            "trip_list", py::overload_cast<>(&mio::abm::Model::get_trip_list),
            [](mio::abm::Model& self, const mio::abm::TripList& list) {
                self.get_trip_list() = list;
            },
            py::return_value_policy::reference_internal)
        .def_property("use_mobility_rules", py::overload_cast<>(&mio::abm::Model::use_mobility_rules, py::const_),
                      py::overload_cast<bool>(&mio::abm::Model::use_mobility_rules))
        .def_readwrite("parameters", &mio::abm::Model::parameters)
        .def_property(
            "testing_strategy", py::overload_cast<>(&mio::abm::Model::get_testing_strategy, py::const_),
            [](mio::abm::Model& self, mio::abm::TestingStrategy strategy) {
                self.get_testing_strategy() = strategy;
            },
            py::return_value_policy::reference_internal);

    pymio::bind_class<mio::abm::Simulation<>, pymio::EnablePickling::Never>(m, "Simulation")
        .def(py::init<mio::abm::TimePoint, size_t>())
        .def("advance",
             &mio::abm::Simulation<>::advance<mio::History<mio::DataWriterToMemory, LogTimePoint, LogLocationIds,
                                                         LogPersonsPerLocationAndInfectionTime, LogAgentIds>>) //AS - welches ist das richtige?
        .def("advance", &mio::abm::Simulation<>::advance<
                            mio::History<mio::DataWriterToMemory, LogTimePoint, LogNewInfectionsAndShedding>>) //AS - welches ist das richtige?
        .def("advance",
             static_cast<void (mio::abm::Simulation<>::*)(mio::abm::TimePoint)>(&mio::abm::Simulation<>::advance),
             py::arg("tmax"))
        .def_property_readonly("model", py::overload_cast<>(&mio::abm::Simulation<>::get_model));


    pymio::bind_class<mio::History<mio::DataWriterToMemory, LogTimePoint, LogLocationIds,
                                   LogPersonsPerLocationAndInfectionTime, LogAgentIds>,
                      pymio::EnablePickling::Never>(m, "History")
        .def(py::init<>())
        .def_property_readonly("log", [](mio::History<mio::DataWriterToMemory, LogTimePoint, LogLocationIds,
                                                      LogPersonsPerLocationAndInfectionTime, LogAgentIds>& self) {
            return self.get_log();
        });

    pymio::bind_class<mio::History<mio::DataWriterToMemory, LogTimePoint, LogNewInfectionsAndShedding>, //AS - brauche ich das? abhängig von dem pybind
                pymio::EnablePickling::Never>(m, "History_sensitivity")
        .def(py::init<>())
        .def_property_readonly(
            "log", [](mio::History<mio::DataWriterToMemory, LogTimePoint, LogNewInfectionsAndShedding>& self) {
                return self.get_log();
            });

    m.attr("__version__") = "dev";

    m.def(
        "set_log_level_warn",
        []() {
            mio::set_log_level(mio::LogLevel::warn);
        },
        py::return_value_policy::reference_internal);
    
    m.def(
        "set_seeds",
        [](mio::abm::Model& model, int seed) {
            auto rng = mio::RandomNumberGenerator();
            rng.seed({static_cast<uint32_t>(seed)});
            model.get_rng() = rng;
        },
        py::return_value_policy::reference_internal);
    
    m.def("initialize_model", &initialize_model, py::return_value_policy::reference_internal);

    
}

PYMIO_IGNORE_VALUE_TYPE(decltype(std::declval<mio::abm::Model>().get_locations()))
PYMIO_IGNORE_VALUE_TYPE(decltype(std::declval<mio::abm::Model>().get_persons()))
