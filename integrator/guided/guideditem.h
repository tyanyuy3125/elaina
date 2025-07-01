#pragma once
#include "core/common.h"
// inherits all items' definition from the wavefront pathtracer
#include "integrator/uniform/workitem.h"
#include "integrator/guided/parameters.h"

ELAINA_NAMESPACE_BEGIN

namespace common2d {

/* Remember to copy these definitions to guideditem.soa whenever changing them. */

struct UniformWalkWorkItem {
	uint itemId;
};

struct GuidedInferenceWorkItem {
	uint itemId;
};

struct WalkRecordItem {
	Color solution;
    Vector2f pos;
    Vector2f dir;
	float dirPdf; // Not involved in differentiation.

	Color thp;
	
	bool isOnNeumannBoundary;
	Vector2f neumannBoundaryNormal;
};

struct GuidedPixelState {
	WalkRecordItem records[MAX_TRAIN_DEPTH + 1];
	uint curDepth{};
};

struct GuidedInput {
	Vector2f pos;
};

struct GuidedOutput {
    Color solution;
	Vector2f dir;
	float dirPdf;

	bool isOnNeumannBoundary;
	Vector2f neumannBoundaryNormal;
};

#pragma warning (push, 0)
#pragma warning (disable: ALL_CODE_ANALYSIS_WARNINGS)
#include "integrator/guided/guided2d_workitem_soa.h"
#pragma warning (pop)

}

namespace common3d {

/* Remember to copy these definitions to guideditem.soa whenever changing them. */

struct UniformWalkWorkItem {
	uint itemId;
};

struct GuidedInferenceWorkItem {
	uint itemId;
};

struct WalkRecordItem {
	Color solution;
    Vector3f pos;
    Vector3f dir;
	float dirPdf;

	Color thp;
	
	bool isOnNeumannBoundary;
	Vector3f neumannBoundaryNormal;
};

struct GuidedPixelState {
	WalkRecordItem records[MAX_TRAIN_DEPTH + 1];
	uint curDepth{};
};

struct GuidedInput {
	Vector3f pos;
};

struct GuidedOutput {
    Color solution;
	Vector3f dir;
	float dirPdf;

	bool isOnNeumannBoundary;
	Vector3f neumannBoundaryNormal;
};

#pragma warning (push, 0)
#pragma warning (disable: ALL_CODE_ANALYSIS_WARNINGS)
#include "integrator/guided/guided3d_workitem_soa.h"
#pragma warning (pop)

}


ELAINA_NAMESPACE_END