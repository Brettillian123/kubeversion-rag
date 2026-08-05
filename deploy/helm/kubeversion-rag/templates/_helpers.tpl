{{- define "kvrag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "kvrag.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "kvrag.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "kvrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "kvrag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "kvrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "kvrag.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "kvrag.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Fail at template time rather than at runtime when the image tag is unset. A pod that
pulls "repository:" is a confusing ImagePullBackOff several minutes into a rollout;
this is an immediate, readable error.
*/}}
{{- define "kvrag.image" -}}
{{- $tag := required "image.tag must be set explicitly (e.g. --set image.tag=$(git rev-parse --short HEAD)); a floating tag makes rollbacks ambiguous" .Values.image.tag -}}
{{- printf "%s/%s/%s:%s" .Values.image.registry .Values.image.repository .component $tag -}}
{{- end -}}

{{- define "kvrag.qdrantUrl" -}}
{{- printf "http://%s-qdrant:%d" (include "kvrag.fullname" .) (int .Values.qdrant.port) -}}
{{- end -}}

{{- define "kvrag.embedUrl" -}}
{{- printf "http://%s-embed:%d" (include "kvrag.fullname" .) (int .Values.embed.port) -}}
{{- end -}}
